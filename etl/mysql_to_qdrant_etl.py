#!/usr/bin/env python3
"""
MySQL to Qdrant ETL Pipeline with BGE-M3 Hybrid Search
=======================================================

Production-grade ETL script to migrate financial data from MySQL to Qdrant
using BGE-M3 embeddings for Hybrid Search (Dense + Sparse vectors).

Features:
- Server-Side Cursor (SSCursor) for memory-efficient streaming of large datasets
- BGE-M3 hybrid embeddings (Dense for semantic + Sparse for lexical matching)
- Modular data sources (Company + Shareholder in single script)
- Batch processing for optimal throughput
- AMD ROCm / NVIDIA CUDA / CPU auto-detection

Author: AI Data Engineer
Data Source: idnfinancials.com
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from dataclasses import dataclass
from typing import Generator, Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import re
import html

import pymysql
from pymysql.cursors import SSCursor
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('etl_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class MySQLConfig:
    """MySQL connection configuration."""
    host: str = os.getenv('MYSQL_HOST', 'localhost')
    port: int = int(os.getenv('MYSQL_PORT', 3306))
    user: str = os.getenv('MYSQL_USER', 'root')
    password: str = os.getenv('MYSQL_PASSWORD', 'root')
    database: str = os.getenv('MYSQL_DATABASE', 'idnrag')
    table: str = os.getenv('MYSQL_TABLE', 'company')
    charset: str = 'utf8mb4'


@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""
    host: str = os.getenv('QDRANT_HOST', 'localhost')
    port: int = int(os.getenv('QDRANT_PORT', 6333))
    collection_name: str = os.getenv('QDRANT_COLLECTION', 'financial_data')
    url: Optional[str] = os.getenv('QDRANT_URL')
    api_key: Optional[str] = os.getenv('QDRANT_API_KEY')
    
    # BGE-M3 produces 1024-dimensional dense vectors
    dense_vector_size: int = 1024


@dataclass
class ETLConfig:
    """ETL process configuration."""
    batch_size: int = int(os.getenv('BATCH_SIZE', 50))
    max_rows: Optional[int] = None  # None = process all rows
    dry_run: bool = False
    include_shareholders: bool = True  # NEW: Include shareholder data


# =============================================================================
# GPU/DEVICE DETECTION
# =============================================================================

def detect_device() -> str:
    """
    Detect available compute device.
    Priority: AMD ROCm (hip) > NVIDIA CUDA > CPU
    
    For AMD RX 6600 XT, PyTorch with ROCm will report 'cuda' but uses HIP backend.
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {device_name}")
            
            # Check if it's AMD (ROCm reports via CUDA interface)
            if 'AMD' in device_name.upper() or 'RADEON' in device_name.upper():
                logger.info("Using AMD GPU via ROCm/HIP backend")
            else:
                logger.info("Using NVIDIA GPU via CUDA backend")
            
            return 'cuda'
        else:
            logger.warning("No GPU detected. Using CPU (slower performance)")
            return 'cpu'
            
    except ImportError:
        logger.warning("PyTorch not installed. Using CPU")
        return 'cpu'


# =============================================================================
# BGE-M3 EMBEDDING MODEL
# =============================================================================

class BGEM3Embedder:
    """
    BGE-M3 Embedding Model wrapper for hybrid search.
    
    Generates both:
    - Dense vectors (1024-dim) for semantic similarity
    - Sparse vectors (token weights) for lexical/keyword matching
    
    The sparse vectors are critical for accurate ticker symbol matching
    (e.g., distinguishing "BBCA" from "BBSA").
    """
    
    def __init__(self, device: str = 'auto'):
        from FlagEmbedding import BGEM3FlagModel
        
        if device == 'auto':
            device = detect_device()
        
        self.device = device
        logger.info(f"Loading BGE-M3 model on device: {device}")
        
        # Load model with appropriate device
        # use_fp16=True for faster inference on GPU
        use_fp16 = (device == 'cuda')
        
        self.model = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=use_fp16,
            device=device
        )
        
        logger.info("BGE-M3 model loaded successfully")
    
    def encode_batch(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Encode a batch of texts into dense and sparse vectors.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Tuple of:
            - dense_vectors: List of 1024-dim float vectors
            - sparse_vectors: List of dicts mapping token_id -> weight
        """
        # BGE-M3 can return both dense and sparse in one call
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False  # We don't need ColBERT vectors
        )
        
        dense_vectors = output['dense_vecs'].tolist()
        
        # Convert sparse vectors to Qdrant-compatible format
        # BGE-M3 returns sparse as a dict of {token_id: weight}
        sparse_vectors = []
        for sparse_dict in output['lexical_weights']:
            # sparse_dict is already {token_id: weight}
            # Convert to proper format
            sparse_vectors.append(dict(sparse_dict))
        
        return dense_vectors, sparse_vectors


# =============================================================================
# MYSQL DATA STREAMING
# =============================================================================

def get_mysql_connection(config: MySQLConfig) -> pymysql.Connection:
    """
    Create MySQL connection with Server-Side Cursor for streaming.
    
    SSCursor is CRITICAL for large datasets - it streams rows from the server
    instead of loading everything into RAM.
    """
    logger.info(f"Connecting to MySQL: {config.host}:{config.port}/{config.database}")
    
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset=config.charset,
        cursorclass=SSCursor,  # Server-Side Cursor for streaming
        connect_timeout=30,
        read_timeout=300,
    )
    
    logger.info("MySQL connection established with SSCursor")
    return connection


def stream_rows(
    config: MySQLConfig,
    max_rows: Optional[int] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream rows from MySQL using Server-Side Cursor.
    
    Yields one row at a time to minimize memory usage.
    
    Args:
        config: MySQL configuration
        max_rows: Optional limit on rows to process
        
    Yields:
        Dict representing each row with column names as keys
    """
    connection = get_mysql_connection(config)
    
    try:
        with connection.cursor() as cursor:
            # Build query - columns from company table for RAG
            query = f"""
                SELECT 
                    id,
                    name,
                    exchange,
                    si_code,
                    summary,
                    description,
                    address,
                    phone,
                    email,
                    website,
                    established,
                    listing,
                    `group`,
                    status,
                    -- Price data
                    close_price,
                    open_price,
                    previous_price,
                    bid_price,
                    offer_price,
                    price_change,
                    percentage_price,
                    day_low,
                    day_high,
                    -- Trading data
                    tradable_volume,
                    tradable_value,
                    total_frequency,
                    capitalization,
                    -- Financial ratios
                    price_earning_ratio,
                    earning_per_share
                FROM {config.table}
                WHERE status = '0'
            """
            
            if max_rows:
                query += f" LIMIT {max_rows}"
            
            logger.info(f"Executing query on table: {config.table}")
            cursor.execute(query)
            
            # Get column names from cursor description
            columns = [col[0] for col in cursor.description]
            
            row_count = 0
            for row in cursor:
                row_count += 1
                # Convert tuple to dict with column names
                yield dict(zip(columns, row))
            
            logger.info(f"Streamed {row_count} rows from MySQL")
            
    finally:
        connection.close()
        logger.info("MySQL connection closed")


def count_rows(config: MySQLConfig) -> int:
    """Get total row count for progress bar."""
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
    )
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {config.table} WHERE status = '0'")
            count = cursor.fetchone()[0]
            return count
    finally:
        connection.close()


# =============================================================================
# SHAREHOLDER DATA FETCHING (NEW)
# =============================================================================

def fetch_all_shareholders(config: MySQLConfig) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all shareholder data from MySQL and group by company.
    
    Uses the query to get latest shareholding data per company.
    Excludes: id, datecreate, datemodified, input_prompt columns.
    
    Returns:
        Dict mapping company code (exchange) to list of shareholders
    """
    logger.info("Fetching shareholder data from MySQL...")
    
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset=config.charset,
    )
    
    shareholders_by_company: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    try:
        with connection.cursor() as cursor:
            # Query to get latest shareholding data per company
            # Excludes: id, datecreate, datemodified, input_prompt
            # Only selects columns that definitely exist
            query = """
                SELECT 
                    s.company,
                    s.holding_date,
                    s.name,
                    s.shares,
                    s.percentage,
                    s.active
                FROM shareholder s
                JOIN (
                    SELECT
                        company,
                        MAX(holding_date) AS max_holding_date
                    FROM shareholder
                    WHERE active = 'Y'
                    GROUP BY company
                ) t
                  ON s.company = t.company
                 AND s.holding_date = t.max_holding_date
                WHERE s.active = 'Y'
                ORDER BY s.company ASC, s.shares DESC
            """
            
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            
            for row in cursor:
                row_dict = dict(zip(columns, row))
                company_code = row_dict.get('company', '')
                if company_code:
                    # Handle VARCHAR to numeric conversion safely
                    shares_str = row_dict.get('shares', '') or '0'
                    percentage_str = row_dict.get('percentage', '') or '0'
                    
                    # Clean and convert shares (remove commas, spaces)
                    try:
                        shares_clean = shares_str.replace(',', '').replace('.', '').strip()
                        shares_val = int(shares_clean) if shares_clean else 0
                    except (ValueError, AttributeError):
                        shares_val = 0
                    
                    # Clean and convert percentage
                    try:
                        percentage_clean = percentage_str.replace(',', '.').strip()
                        percentage_val = float(percentage_clean) if percentage_clean else 0.0
                    except (ValueError, AttributeError):
                        percentage_val = 0.0
                    
                    shareholders_by_company[company_code].append({
                        'name': row_dict.get('name', ''),
                        'shares': shares_val,
                        'percentage': percentage_val,
                        'holding_date': str(row_dict.get('holding_date', '')) if row_dict.get('holding_date') else None,
                    })
            
            total_shareholders = sum(len(v) for v in shareholders_by_company.values())
            logger.info(f"Fetched {total_shareholders} shareholders for {len(shareholders_by_company)} companies")
            
    finally:
        connection.close()
    
    return dict(shareholders_by_company)


def format_shareholders_for_text(shareholders: List[Dict[str, Any]], max_display: int = 10) -> str:
    """
    Format shareholder data into natural language for embedding.
    
    Args:
        shareholders: List of shareholder dicts
        max_display: Max shareholders to include in text (for embedding brevity)
        
    Returns:
        Natural language description of shareholders
    """
    if not shareholders:
        return ""
    
    parts = []
    parts.append("Pemegang Saham Utama:")
    
    # Include all shareholder names for better semantic search (reverse lookup)
    for i, sh in enumerate(shareholders[:max_display]):
        name = sh.get('name', 'Unknown')
        percentage = sh.get('percentage', 0)
        shares = sh.get('shares', 0)
        
        # Format shares with thousands separator
        shares_formatted = f"{shares:,}" if shares else "N/A"
        
        line = f"- {name}: {percentage:.2f}% ({shares_formatted} lembar saham)"
        parts.append(line)
    
    if len(shareholders) > max_display:
        remaining = len(shareholders) - max_display
        # Include remaining shareholder NAMES for search (without full details)
        remaining_names = [sh.get('name', '') for sh in shareholders[max_display:] if sh.get('name')]
        if remaining_names:
            parts.append(f"- Pemegang saham lainnya: {', '.join(remaining_names[:10])}")
            if len(remaining_names) > 10:
                parts.append(f"  ...dan {len(remaining_names) - 10} pemegang saham lainnya")
    
    return "\n".join(parts)


# =============================================================================
# DIVIDEND DATA FETCHING
# =============================================================================

def fetch_all_dividends(config: MySQLConfig) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch dividend data from MySQL and group by company.
    
    Gets 5 latest dividends per company with all relevant fields.
    Calculates dividend_yield = (dps / close_price) * 100
    
    Returns:
        Dict mapping company code to list of dividend records
    """
    logger.info("Fetching dividend data from MySQL...")
    
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset=config.charset,
    )
    
    dividends_by_company: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    try:
        with connection.cursor() as cursor:
            # Query to get dividends ordered by year (latest first)
            # Using ROW_NUMBER to get top 5 per company
            query = """
                SELECT 
                    company_code,
                    year,
                    cash_dividend,
                    dividend_shares,
                    close_price,
                    type,
                    cum_date,
                    ex_date,
                    recording_date,
                    payment_date,
                    payout_ratio
                FROM idn_dividend
                WHERE company_code IS NOT NULL AND company_code != ''
                ORDER BY company_code ASC, year DESC
            """
            
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            
            # Track count per company to limit to 5
            company_counts: Dict[str, int] = defaultdict(int)
            
            for row in cursor:
                row_dict = dict(zip(columns, row))
                company_code = row_dict.get('company_code', '')
                
                # Skip if already have 5 for this company
                if company_counts[company_code] >= 5:
                    continue
                
                company_counts[company_code] += 1
                
                # Parse numeric values safely
                try:
                    dps_str = str(row_dict.get('dividend_shares', '') or '0')
                    dps = float(dps_str.replace(',', '').strip()) if dps_str else 0.0
                except (ValueError, AttributeError):
                    dps = 0.0
                
                try:
                    close_str = str(row_dict.get('close_price', '') or '0')
                    close_price = float(close_str.replace(',', '').strip()) if close_str else 0.0
                except (ValueError, AttributeError):
                    close_price = 0.0
                
                try:
                    cash_str = str(row_dict.get('cash_dividend', '') or '0')
                    cash_dividend = float(cash_str.replace(',', '').strip()) if cash_str else 0.0
                except (ValueError, AttributeError):
                    cash_dividend = 0.0
                
                # Calculate dividend yield
                dividend_yield = (dps / close_price * 100) if close_price > 0 else 0.0
                
                # Format dates
                def format_date(d):
                    return str(d) if d else None
                
                dividends_by_company[company_code].append({
                    'year': int(row_dict.get('year', 0)) if row_dict.get('year') else None,
                    'total_cash_dividend': cash_dividend,
                    'dividend_per_share': dps,
                    'close_price': close_price,
                    'dividend_yield': round(dividend_yield, 2),
                    'type': row_dict.get('type', ''),
                    'cum_date': format_date(row_dict.get('cum_date')),
                    'ex_date': format_date(row_dict.get('ex_date')),
                    'recording_date': format_date(row_dict.get('recording_date')),
                    'payment_date': format_date(row_dict.get('payment_date')),
                    'payout_ratio': row_dict.get('payout_ratio', ''),
                })
            
            total_dividends = sum(len(v) for v in dividends_by_company.values())
            logger.info(f"Fetched {total_dividends} dividends for {len(dividends_by_company)} companies")
            
    finally:
        connection.close()
    
    return dict(dividends_by_company)


def format_dividends_for_text(dividends: List[Dict[str, Any]], max_display: int = 3) -> str:
    """
    Format dividend data into natural language for embedding.
    
    Args:
        dividends: List of dividend dicts
        max_display: Max dividends to include in text
        
    Returns:
        Natural language description of dividends
    """
    if not dividends:
        return ""
    
    parts = []
    parts.append("Riwayat Dividen:")
    
    for div in dividends[:max_display]:
        year = div.get('year', 'N/A')
        dps = div.get('dividend_per_share', 0)
        div_yield = div.get('dividend_yield', 0)
        div_type = div.get('type', '')
        
        # Use clear, standard naming for AI understanding
        line = f"- Tahun {year}: Dividend Per Share Rp {dps:,.0f} (Dividend Yield: {div_yield:.2f}%)"
        if div_type:
            line += f" [{div_type}]"
        parts.append(line)
    
    if len(dividends) > max_display:
        parts.append(f"- ...dan {len(dividends) - max_display} dividen lainnya")
    
    return "\n".join(parts)


# =============================================================================
# FINANCIAL DATA FETCHING (ks_financial + ks_financial_bank)
# =============================================================================

def _parse_numeric(value: Any) -> Optional[float]:
    """Safely parse a numeric value from VARCHAR."""
    if value is None or value == '':
        return None
    try:
        # Remove commas and convert
        clean = str(value).replace(',', '').strip()
        return float(clean) if clean else None
    except (ValueError, TypeError):
        return None


def _parse_period_key(asof_date: str) -> tuple:
    """Parse period string like 'Q4/2024' into sortable tuple (year, quarter)."""
    if not asof_date:
        return (0, 0)
    try:
        parts = asof_date.split('/')
        if len(parts) == 2:
            quarter = int(parts[0].replace('Q', ''))
            year = int(parts[1])
            return (year, quarter)
    except:
        pass
    return (0, 0)


def fetch_all_financials(config: MySQLConfig) -> Dict[str, Dict[str, Any]]:
    """
    Fetch financial data from ks_financial and ks_financial_bank.
    
    Returns dict mapping company code to financial data including:
    - Latest quarter data
    - 4 quarter history
    - YoY comparison
    - Banking-specific metrics if applicable
    
    Returns:
        Dict mapping company code to financial data dict
    """
    logger.info("Fetching financial data from MySQL...")
    
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset=config.charset,
    )
    
    financials_by_company: Dict[str, Dict[str, Any]] = {}
    
    try:
        with connection.cursor() as cursor:
            # ============================================
            # FETCH NON-BANK FINANCIAL DATA (ks_financial)
            # ============================================
            query_financial = """
                SELECT 
                    exchange,
                    asof_date,
                    revenue,
                    cogs,
                    gross_profit,
                    gross_margin,
                    operating_expense,
                    operating_profit,
                    operating_margin,
                    net_profit,
                    net_margin,
                    total_shares,
                    eps,
                    current_assets,
                    ar_lancar,
                    ar_lancarp,
                    ar_1m,
                    ar_1mp,
                    ar_2m,
                    ar_2mp,
                    ar_3m,
                    ar_3mp,
                    ar_4m,
                    ar_4mp,
                    ar_g4m,
                    ar_g4mp,
                    ar_total,
                    non_current_assets,
                    tax_assets,
                    total_assets,
                    current_liabilities,
                    non_current_liabilities,
                    tax_liabilities,
                    total_liabilities,
                    total_equity,
                    minority_interest,
                    roa,
                    roe,
                    net_cash,
                    cash_on_hand,
                    currency,
                    idr_rate
                FROM ks_financial
                WHERE exchange IS NOT NULL AND exchange != ''
                ORDER BY exchange ASC, 
                         CAST(SUBSTRING_INDEX(asof_date, '/', -1) AS UNSIGNED) DESC,
                         CAST(REPLACE(SUBSTRING_INDEX(asof_date, '/', 1), 'Q', '') AS UNSIGNED) DESC
            """
            
            cursor.execute(query_financial)
            columns = [col[0] for col in cursor.description]
            
            # Group by company, keep latest 8 quarters (for YoY comparison)
            company_records: Dict[str, List[Dict]] = defaultdict(list)
            
            for row in cursor:
                row_dict = dict(zip(columns, row))
                company = row_dict.get('exchange', '')
                if company and len(company_records[company]) < 8:
                    company_records[company].append(row_dict)
            
            logger.info(f"Fetched financial data for {len(company_records)} non-bank companies")
            
            # ============================================
            # FETCH BANK FINANCIAL DATA (ks_financial_bank)
            # ============================================
            query_bank = """
                SELECT 
                    exchange,
                    asof_date,
                    net_interest_income,
                    operating_expense,
                    operating_profit,
                    net_profit,
                    total_shares,
                    eps,
                    total_assets,
                    total_liabilities,
                    total_equity,
                    minority_interest,
                    car,
                    ldr,
                    npl,
                    roe,
                    roa,
                    net_cash,
                    cash_on_hand,
                    currency
                FROM ks_financial_bank
                WHERE exchange IS NOT NULL AND exchange != ''
                ORDER BY exchange ASC,
                         CAST(SUBSTRING_INDEX(asof_date, '/', -1) AS UNSIGNED) DESC,
                         CAST(REPLACE(SUBSTRING_INDEX(asof_date, '/', 1), 'Q', '') AS UNSIGNED) DESC
            """
            
            cursor.execute(query_bank)
            columns_bank = [col[0] for col in cursor.description]
            
            bank_records: Dict[str, List[Dict]] = defaultdict(list)
            
            for row in cursor:
                row_dict = dict(zip(columns_bank, row))
                company = row_dict.get('exchange', '')
                if company and len(bank_records[company]) < 8:
                    bank_records[company].append(row_dict)
            
            logger.info(f"Fetched financial data for {len(bank_records)} bank companies")
            
            # ============================================
            # PROCESS AND STRUCTURE DATA
            # ============================================
            
            # Process non-bank companies
            for company, records in company_records.items():
                if not records:
                    continue
                
                latest = records[0]
                financials_by_company[company] = _process_financial_records(
                    records, is_bank=False
                )
            
            # Process bank companies (will override non-bank if exists)
            for company, records in bank_records.items():
                if not records:
                    continue
                
                financials_by_company[company] = _process_financial_records(
                    records, is_bank=True
                )
            
            logger.info(f"Processed financial data for {len(financials_by_company)} total companies")
            
    finally:
        connection.close()
    
    return financials_by_company


def _process_financial_records(records: List[Dict], is_bank: bool) -> Dict[str, Any]:
    """
    Process financial records into structured data.
    
    Args:
        records: List of financial records (sorted by date descending)
        is_bank: Whether this is bank data
        
    Returns:
        Structured financial data dict
    """
    if not records:
        return {}
    
    latest = records[0]
    
    # Find same quarter from previous year for YoY comparison
    latest_period = latest.get('asof_date', '')
    latest_year, latest_quarter = _parse_period_key(latest_period)
    
    prev_year_record = None
    for rec in records:
        rec_year, rec_quarter = _parse_period_key(rec.get('asof_date', ''))
        if rec_year == latest_year - 1 and rec_quarter == latest_quarter:
            prev_year_record = rec
            break
    
    # Build 4-quarter history
    history = []
    for rec in records[:4]:
        hist_item = {
            'period': rec.get('asof_date', ''),
            'net_profit': _parse_numeric(rec.get('net_profit')),
            'roe': _parse_numeric(rec.get('roe')),
            'roa': _parse_numeric(rec.get('roa')),
        }
        
        if not is_bank:
            hist_item['revenue'] = _parse_numeric(rec.get('revenue'))
            hist_item['gross_margin'] = _parse_numeric(rec.get('gross_margin'))
            hist_item['operating_margin'] = _parse_numeric(rec.get('operating_margin'))
            hist_item['net_margin'] = _parse_numeric(rec.get('net_margin'))
        
        history.append(hist_item)
    
    # Calculate YoY growth
    latest_net_profit = _parse_numeric(latest.get('net_profit'))
    prev_net_profit = _parse_numeric(prev_year_record.get('net_profit')) if prev_year_record else None
    
    net_profit_yoy_growth = None
    if latest_net_profit and prev_net_profit and prev_net_profit != 0:
        net_profit_yoy_growth = ((latest_net_profit - prev_net_profit) / abs(prev_net_profit)) * 100
    
    revenue_yoy_growth = None
    if not is_bank:
        latest_revenue = _parse_numeric(latest.get('revenue'))
        prev_revenue = _parse_numeric(prev_year_record.get('revenue')) if prev_year_record else None
        if latest_revenue and prev_revenue and prev_revenue != 0:
            revenue_yoy_growth = ((latest_revenue - prev_revenue) / abs(prev_revenue)) * 100
    
    # Build result
    result = {
        'is_bank': is_bank,
        'financial_period': latest_period,
        
        # Latest values
        'net_profit': _parse_numeric(latest.get('net_profit')),
        'total_shares': _parse_numeric(latest.get('total_shares')),
        'eps': _parse_numeric(latest.get('eps')),
        'total_assets': _parse_numeric(latest.get('total_assets')),
        'total_liabilities': _parse_numeric(latest.get('total_liabilities')),
        'total_equity': _parse_numeric(latest.get('total_equity')),
        'roe': _parse_numeric(latest.get('roe')),
        'roa': _parse_numeric(latest.get('roa')),
        'net_cash': _parse_numeric(latest.get('net_cash')),
        'cash_on_hand': _parse_numeric(latest.get('cash_on_hand')),
        
        # YoY Growth
        'net_profit_yoy_growth': round(net_profit_yoy_growth, 2) if net_profit_yoy_growth else None,
        
        # Previous year values (for comparison)
        'net_profit_prev_year': _parse_numeric(prev_year_record.get('net_profit')) if prev_year_record else None,
        'roe_prev_year': _parse_numeric(prev_year_record.get('roe')) if prev_year_record else None,
        
        # History
        'financial_history': history,
    }
    
    if is_bank:
        # Banking-specific metrics
        # Calculate ROE and ROA since ks_financial_bank doesn't have them
        net_profit = _parse_numeric(latest.get('net_profit'))
        total_assets = _parse_numeric(latest.get('total_assets'))
        total_equity = _parse_numeric(latest.get('total_equity'))
        
        calculated_roe = None
        calculated_roa = None
        
        if net_profit and total_equity and total_equity != 0:
            calculated_roe = round((net_profit / total_equity) * 100, 2)
        
        if net_profit and total_assets and total_assets != 0:
            calculated_roa = round((net_profit / total_assets) * 100, 2)
        
        # Update ROE/ROA with calculated values
        result['roe'] = calculated_roe
        result['roa'] = calculated_roa
        
        result.update({
            'net_interest_income': _parse_numeric(latest.get('net_interest_income')),
            'operating_expense': _parse_numeric(latest.get('operating_expense')),
            'operating_profit': _parse_numeric(latest.get('operating_profit')),
        })
    else:
        # Non-bank metrics
        # Calculate ROE/ROA if not present
        net_profit = _parse_numeric(latest.get('net_profit'))
        total_assets = _parse_numeric(latest.get('total_assets'))
        total_equity = _parse_numeric(latest.get('total_equity'))
        
        if not result['roe'] and net_profit and total_equity and total_equity != 0:
            result['roe'] = round((net_profit / total_equity) * 100, 2)
        
        if not result['roa'] and net_profit and total_assets and total_assets != 0:
            result['roa'] = round((net_profit / total_assets) * 100, 2)
        
        result.update({
            'revenue': _parse_numeric(latest.get('revenue')),
            'cogs': _parse_numeric(latest.get('cogs')),
            'gross_profit': _parse_numeric(latest.get('gross_profit')),
            'gross_margin': _parse_numeric(latest.get('gross_margin')),
            'operating_expense': _parse_numeric(latest.get('operating_expense')),
            'operating_profit': _parse_numeric(latest.get('operating_profit')),
            'operating_margin': _parse_numeric(latest.get('operating_margin')),
            'net_margin': _parse_numeric(latest.get('net_margin')),
            'current_assets': _parse_numeric(latest.get('current_assets')),
            'non_current_assets': _parse_numeric(latest.get('non_current_assets')),
            'current_liabilities': _parse_numeric(latest.get('current_liabilities')),
            'non_current_liabilities': _parse_numeric(latest.get('non_current_liabilities')),
            'revenue_yoy_growth': round(revenue_yoy_growth, 2) if revenue_yoy_growth else None,
            # Accounts Receivable Aging
            'ar_lancar': _parse_numeric(latest.get('ar_lancar')),
            'ar_lancarp': _parse_numeric(latest.get('ar_lancarp')),
            'ar_1m': _parse_numeric(latest.get('ar_1m')),
            'ar_1mp': _parse_numeric(latest.get('ar_1mp')),
            'ar_2m': _parse_numeric(latest.get('ar_2m')),
            'ar_2mp': _parse_numeric(latest.get('ar_2mp')),
            'ar_3m': _parse_numeric(latest.get('ar_3m')),
            'ar_3mp': _parse_numeric(latest.get('ar_3mp')),
            'ar_4m': _parse_numeric(latest.get('ar_4m')),
            'ar_4mp': _parse_numeric(latest.get('ar_4mp')),
            'ar_g4m': _parse_numeric(latest.get('ar_g4m')),
            'ar_g4mp': _parse_numeric(latest.get('ar_g4mp')),
            'ar_total': _parse_numeric(latest.get('ar_total')),
            # Tax
            'tax_assets': _parse_numeric(latest.get('tax_assets')),
            'tax_liabilities': _parse_numeric(latest.get('tax_liabilities')),
            # Currency
            'idr_rate': _parse_numeric(latest.get('idr_rate')),
            'currency': latest.get('currency'),
        })
    
    return result


def format_financials_for_text(financials: Dict[str, Any]) -> str:
    """
    Format financial data into natural language for embedding.
    
    Creates rich text that helps semantic search find financial queries.
    
    Args:
        financials: Financial data dict from fetch_all_financials
        
    Returns:
        Natural language description of financials
    """
    if not financials:
        return ""
    
    parts = []
    period = financials.get('financial_period', 'N/A')
    is_bank = financials.get('is_bank', False)
    
    # Header
    parts.append(f"📊 Laporan Keuangan ({period}):")
    
    # Main metrics
    net_profit = financials.get('net_profit')
    roe = financials.get('roe')
    roa = financials.get('roa')
    
    if net_profit:
        net_profit_fmt = _format_rupiah(net_profit)
        yoy = financials.get('net_profit_yoy_growth')
        yoy_str = f" ({yoy:+.1f}% YoY)" if yoy else ""
        parts.append(f"- Net Profit: {net_profit_fmt}{yoy_str}")
    
    if not is_bank:
        # Non-bank specific
        revenue = financials.get('revenue')
        if revenue:
            revenue_fmt = _format_rupiah(revenue)
            rev_yoy = financials.get('revenue_yoy_growth')
            rev_yoy_str = f" ({rev_yoy:+.1f}% YoY)" if rev_yoy else ""
            parts.append(f"- Revenue: {revenue_fmt}{rev_yoy_str}")
        
        gross_margin = financials.get('gross_margin')
        if gross_margin:
            parts.append(f"- Gross Margin: {gross_margin:.1f}%")
        
        net_margin = financials.get('net_margin')
        if net_margin:
            parts.append(f"- Net Margin: {net_margin:.1f}%")
    
    if roe:
        parts.append(f"- ROE: {roe:.2f}%")
    if roa:
        parts.append(f"- ROA: {roa:.2f}%")
    
    # Banking-specific - Net Interest Income
    if is_bank:
        nii = financials.get('net_interest_income')
        if nii:
            parts.append(f"- Net Interest Income: {_format_rupiah(nii)}")
    
    # 4-Quarter Trend
    history = financials.get('financial_history', [])
    if len(history) >= 2:
        parts.append(f"\n📈 Trend {len(history)} Quarter:")
        for hist in history:
            hist_period = hist.get('period', '')
            hist_np = hist.get('net_profit')
            hist_roe = hist.get('roe')
            
            line_parts = [f"- {hist_period}:"]
            if hist_np:
                line_parts.append(f"Net Profit {_format_rupiah(hist_np)}")
            if hist_roe:
                line_parts.append(f"ROE {hist_roe:.1f}%")
            
            if len(line_parts) > 1:
                parts.append(" | ".join(line_parts))
    
    return "\n".join(parts)


def _format_rupiah(value: float) -> str:
    """Format large numbers in Rupiah with T/M suffix."""
    if value is None:
        return "N/A"
    
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1_000_000_000_000:
        return f"{sign}Rp {abs_value / 1_000_000_000_000:.1f} T"
    elif abs_value >= 1_000_000_000:
        return f"{sign}Rp {abs_value / 1_000_000_000:.1f} M"
    elif abs_value >= 1_000_000:
        return f"{sign}Rp {abs_value / 1_000_000:.1f} Jt"
    else:
        return f"{sign}Rp {abs_value:,.0f}"


# =============================================================================
# ROW SERIALIZATION (Tabular → Natural Language)
# =============================================================================

def clean_html(raw_html: str) -> str:
    """
    Clean HTML content for embedding.
    
    Removes HTML tags, entities, and extra whitespace.
    """
    if not raw_html:
        return ""
    
    # Decode HTML entities
    text = html.unescape(raw_html)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def _extract_year(date_val) -> Optional[int]:
    """Extract year as integer from date value for Range filtering."""
    if not date_val:
        return None
    try:
        date_str = str(date_val)
        # Try to extract year from date string (YYYY-MM-DD or similar)
        match = re.match(r'(\d{4})', date_str)
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None


def serialize_row(
    row: Dict[str, Any], 
    shareholders: Optional[List[Dict[str, Any]]] = None,
    dividends: Optional[List[Dict[str, Any]]] = None,
    financials: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convert company data into natural language for embedding.
    
    Optimized for RAG queries about Indonesian listed companies.
    Includes listing date for date-based semantic search.
    Now includes shareholder, dividend, and financial statement data.
    
    Example:
        Input:  {'name': 'PT. Eagle High Plantations Tbk', 'exchange': 'BWPT', ...}
        Output: "PT. Eagle High Plantations Tbk (BWPT)
                 PT Eagle High Plantations Tbk was initiated in 2000...
                 Pemegang Saham Utama: ...
                 Riwayat Dividen: ...
                 📊 Laporan Keuangan: ..."
    """
    name = row.get('name', 'Unknown Company')
    exchange = row.get('exchange', '')
    summary = row.get('summary', '') or ''
    description = clean_html(row.get('description', '') or '')
    si_code = row.get('si_code', '')
    address = row.get('address', '') or ''
    website = row.get('website', '') or ''
    established = row.get('established', '') or ''
    listing = row.get('listing', '') or ''  # Added listing date
    group = row.get('group', '') or ''
    
    # Build natural language description
    parts = []
    
    # Company identity
    parts.append(f"{name} ({exchange})")
    
    # Summary
    if summary:
        parts.append(summary)
    
    # Full description (cleaned from HTML)
    if description:
        parts.append(description)
    
    # Additional context - include dates for semantic search
    context_parts = []
    if established:
        context_parts.append(f"Established: {established}")
    if listing:
        # Include listing date with year for better semantic matching
        context_parts.append(f"Listing Date: {listing}")
    if group:
        context_parts.append(f"Group: {group}")
    if address:
        # Clean address for readability
        clean_address = address.replace('\n', ', ').replace('\r', '')
        context_parts.append(f"Location: {clean_address}")
    if website:
        context_parts.append(f"Website: {website}")
    
    if context_parts:
        parts.append(" | ".join(context_parts))
    
    # Add financial statement data for semantic search
    if financials:
        financial_text = format_financials_for_text(financials)
        if financial_text:
            parts.append(financial_text)
    
    # Add shareholder information for semantic search
    if shareholders:
        shareholder_text = format_shareholders_for_text(shareholders)
        if shareholder_text:
            parts.append(shareholder_text)
    
    # Add dividend information for semantic search
    if dividends:
        dividend_text = format_dividends_for_text(dividends)
        if dividend_text:
            parts.append(dividend_text)
    
    return "\n\n".join(parts)


# =============================================================================
# QDRANT COLLECTION SETUP
# =============================================================================

def setup_qdrant_collection(client: QdrantClient, config: QdrantConfig) -> None:
    """
    Create or verify Qdrant collection with hybrid search configuration.
    
    Sets up:
    - Dense vector config (1024-dim, cosine similarity)
    - Sparse vector config (for lexical matching)
    
    Args:
        client: Qdrant client instance
        config: Qdrant configuration
    """
    collection_name = config.collection_name
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name in collection_names:
        logger.info(f"Collection '{collection_name}' already exists")
        
        # Optionally recreate (uncomment if needed)
        # logger.warning(f"Deleting existing collection '{collection_name}'")
        # client.delete_collection(collection_name)
        return
    
    logger.info(f"Creating collection '{collection_name}' with hybrid search config")
    
    # Create collection with both dense and sparse vector support
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            # Dense vector for semantic similarity search
            "dense": VectorParams(
                size=config.dense_vector_size,  # BGE-M3 = 1024
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            # Sparse vector for lexical/keyword matching
            # Critical for exact ticker symbol matching (BBCA vs BBSA)
            "sparse": SparseVectorParams()
        },
    )
    
    logger.info(f"Collection '{collection_name}' created successfully")
    
    # Create payload indices for filterable fields
    logger.info("Creating payload indices for filtering...")
    
    filterable_fields = [
        ("exchange", "keyword"),      # Stock code like BBCA, BBRI
        ("si_code", "keyword"),       # Sector code like D, G
        ("is_bank", "bool"),          # Bank vs non-bank flag
    ]
    
    for field_name, field_type in filterable_fields:
        try:
            if field_type == "keyword":
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema="keyword"
                )
            elif field_type == "bool":
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema="bool"
                )
            logger.info(f"  ✓ Index created: {field_name} ({field_type})")
        except Exception as e:
            logger.warning(f"  ⚠ Could not create index for {field_name}: {e}")


# =============================================================================
# BATCH PROCESSING AND UPSERT
# =============================================================================

def process_and_upsert_batch(
    batch: List[Dict[str, Any]],
    embedder: BGEM3Embedder,
    client: QdrantClient,
    config: QdrantConfig,
    start_id: int,
    shareholders_map: Dict[str, List[Dict[str, Any]]],
    dividends_map: Dict[str, List[Dict[str, Any]]],
    financials_map: Dict[str, Dict[str, Any]] = None,
    dry_run: bool = False
) -> int:
    """
    Process a batch of rows: serialize, embed, and upsert to Qdrant.
    
    Args:
        batch: List of row dictionaries from MySQL
        embedder: BGE-M3 embedding model
        client: Qdrant client
        config: Qdrant configuration
        start_id: Starting point ID for this batch
        shareholders_map: Dict mapping company code to shareholder list
        dividends_map: Dict mapping company code to dividend list
        financials_map: Dict mapping company code to financial data
        dry_run: If True, skip actual upsert
        
    Returns:
        Number of points upserted
    """
    if not batch:
        return 0
    
    if financials_map is None:
        financials_map = {}
    
    # Step 1: Serialize rows to natural language (with shareholders, dividends, and financials)
    texts = []
    for row in batch:
        exchange = row.get('exchange', '')
        company_shareholders = shareholders_map.get(exchange, [])
        company_dividends = dividends_map.get(exchange, [])
        company_financials = financials_map.get(exchange, {})
        text = serialize_row(row, company_shareholders, company_dividends, company_financials)
        texts.append(text)
    
    # Step 2: Generate embeddings (both dense and sparse)
    dense_vectors, sparse_vectors = embedder.encode_batch(texts)
    
    # Step 3: Prepare points for Qdrant
    points = []
    for i, (row, text, dense_vec, sparse_vec) in enumerate(
        zip(batch, texts, dense_vectors, sparse_vectors)
    ):
        point_id = start_id + i
        exchange = row.get('exchange', '')
        
        # Convert sparse vector to Qdrant format
        # sparse_vec is {token_id: weight}
        sparse_indices = list(sparse_vec.keys())
        sparse_values = list(sparse_vec.values())
        
        # Get shareholders for this company
        company_shareholders = shareholders_map.get(exchange, [])
        # Get dividends for this company
        company_dividends = dividends_map.get(exchange, [])
        # Get financials for this company
        company_financials = financials_map.get(exchange, {})
        
        # Extract latest dividend info for quick-access sorting/filtering
        latest_dividend = company_dividends[0] if company_dividends else {}
        latest_dividend_yield = latest_dividend.get('dividend_yield', 0) if latest_dividend else 0
        latest_dps = latest_dividend.get('dividend_per_share', 0) if latest_dividend else 0
        latest_dividend_year = latest_dividend.get('year') if latest_dividend else None
        
        # Calculate P/BV ratio if we have the data
        close_price = float(row.get('close_price')) if row.get('close_price') else None
        total_shares = company_financials.get('total_shares')
        total_equity = company_financials.get('total_equity')
        
        pbv_ratio = None
        if close_price and total_shares and total_equity and total_equity > 0:
            book_value_per_share = total_equity / total_shares
            if book_value_per_share > 0:
                pbv_ratio = round(close_price / book_value_per_share, 2)
        
        # Calculate DER ratio
        total_liabilities = company_financials.get('total_liabilities')
        der_ratio = None
        if total_liabilities and total_equity and total_equity > 0:
            der_ratio = round(total_liabilities / total_equity, 2)
        
        point = PointStruct(
            id=point_id,
            vector={
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                )
            },
            payload={
                # Company identity for filtering
                "exchange": exchange,
                "name": row.get('name'),
                "si_code": row.get('si_code'),
                "group": row.get('group'),
                # Contact info
                "address": row.get('address'),
                "phone": row.get('phone'),
                "email": row.get('email'),
                "website": row.get('website'),
                # Dates (string for display)
                "established": str(row.get('established')) if row.get('established') else None,
                "listing": str(row.get('listing')) if row.get('listing') else None,
                # Listing year as INTEGER for Range filtering
                "listing_year": _extract_year(row.get('listing')),
                # Price data (numeric for filtering)
                "close_price": close_price,
                "open_price": float(row.get('open_price')) if row.get('open_price') else None,
                "previous_price": float(row.get('previous_price')) if row.get('previous_price') else None,
                "bid_price": float(row.get('bid_price')) if row.get('bid_price') else None,
                "offer_price": float(row.get('offer_price')) if row.get('offer_price') else None,
                "price_change": float(row.get('price_change')) if row.get('price_change') else None,
                "percentage_price": float(row.get('percentage_price')) if row.get('percentage_price') else None,
                "day_low": float(row.get('day_low')) if row.get('day_low') else None,
                "day_high": float(row.get('day_high')) if row.get('day_high') else None,
                # Trading data (numeric for filtering)
                "tradable_volume": int(row.get('tradable_volume')) if row.get('tradable_volume') else None,
                "tradable_value": float(row.get('tradable_value')) if row.get('tradable_value') else None,
                "total_frequency": int(row.get('total_frequency')) if row.get('total_frequency') else None,
                "capitalization": float(row.get('capitalization')) if row.get('capitalization') else None,
                # Financial ratios from company table (numeric for filtering)
                "price_earning_ratio": float(row.get('price_earning_ratio')) if row.get('price_earning_ratio') else None,
                "earning_per_share": float(row.get('earning_per_share')) if row.get('earning_per_share') else None,
                
                # === NEW: FINANCIAL STATEMENT DATA ===
                "financial_period": company_financials.get('financial_period'),
                "is_bank": company_financials.get('is_bank', False),
                
                # Profitability
                "revenue": company_financials.get('revenue'),
                "gross_profit": company_financials.get('gross_profit'),
                "operating_profit": company_financials.get('operating_profit'),
                "net_profit": company_financials.get('net_profit'),
                
                # Margins
                "gross_margin": company_financials.get('gross_margin'),
                "operating_margin": company_financials.get('operating_margin'),
                "net_margin": company_financials.get('net_margin'),
                
                # Return ratios
                "roe": company_financials.get('roe'),
                "roa": company_financials.get('roa'),
                
                # Balance sheet
                "total_assets": company_financials.get('total_assets'),
                "total_liabilities": company_financials.get('total_liabilities'),
                "total_equity": company_financials.get('total_equity'),
                "current_assets": company_financials.get('current_assets'),
                "current_liabilities": company_financials.get('current_liabilities'),
                
                # Calculated ratios
                "pbv_ratio": pbv_ratio,
                "der_ratio": der_ratio,
                
                # Cash
                "net_cash": company_financials.get('net_cash'),
                "cash_on_hand": company_financials.get('cash_on_hand'),
                
                # Banking-specific metrics (net_interest_income only)
                "net_interest_income": company_financials.get('net_interest_income'),
                
                # YoY Growth metrics
                "net_profit_yoy_growth": company_financials.get('net_profit_yoy_growth'),
                "revenue_yoy_growth": company_financials.get('revenue_yoy_growth'),
                "net_profit_prev_year": company_financials.get('net_profit_prev_year'),
                "roe_prev_year": company_financials.get('roe_prev_year'),
                
                # Quarterly history (for trend analysis)
                "financial_history": company_financials.get('financial_history', []),
                
                # === END FINANCIAL DATA ===
                
                # Shareholder data (structured for filtering)
                "shareholders": company_shareholders,
                "top_shareholder": company_shareholders[0].get('name') if company_shareholders else None,
                "top_shareholder_percentage": company_shareholders[0].get('percentage') if company_shareholders else None,
                "total_shareholders": len(company_shareholders),
                # Dividend data (structured + quick-access fields)
                "dividends": company_dividends,  # Full dividend history (5 years)
                "latest_dividend_yield": latest_dividend_yield,  # For sorting: yield tertinggi
                "latest_dividend_per_share": latest_dps,  # For sorting: DPS terbesar
                "latest_dividend_year": latest_dividend_year,  # Year of latest dividend
                "has_dividend": len(company_dividends) > 0,  # Boolean filter
                # Source
                "source": "idnfinancials.com",
                # Serialized text for reference
                "text": text,
                # Metadata
                "ingested_at": datetime.utcnow().isoformat(),
                "original_id": row.get('id'),
            }
        )
        points.append(point)
    
    # Step 4: Upsert to Qdrant
    if not dry_run:
        client.upsert(
            collection_name=config.collection_name,
            points=points,
        )
    else:
        logger.info(f"[DRY RUN] Would upsert {len(points)} points")
    
    return len(points)


# =============================================================================
# MAIN ETL ORCHESTRATION
# =============================================================================

def run_etl(
    mysql_config: MySQLConfig,
    qdrant_config: QdrantConfig,
    etl_config: ETLConfig
) -> Dict[str, Any]:
    """
    Main ETL orchestration function.
    
    Streams data from MySQL, generates hybrid embeddings, and upserts to Qdrant
    in batches for optimal performance.
    
    Args:
        mysql_config: MySQL connection settings
        qdrant_config: Qdrant connection settings
        etl_config: ETL process settings
        
    Returns:
        Dictionary with ETL statistics
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Starting MySQL to Qdrant ETL Pipeline")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing BGE-M3 embedding model...")
    embedder = BGEM3Embedder(device='auto')
    
    # Initialize Qdrant clients (support dual-target: Cloud + Local)
    clients = []
    
    # Primary: Cloud (if configured)
    if qdrant_config.url:
        logger.info(f"Connecting to Qdrant Cloud: {qdrant_config.url}")
        cloud_client = QdrantClient(
            url=qdrant_config.url,
            api_key=qdrant_config.api_key,
            timeout=60
        )
        clients.append(("Cloud", cloud_client))
    
    # Secondary: Local (always try if host is configured)
    if qdrant_config.host:
        try:
            logger.info(f"Connecting to Qdrant Local: {qdrant_config.host}:{qdrant_config.port}")
            local_client = QdrantClient(
                host=qdrant_config.host,
                port=qdrant_config.port,
                timeout=10
            )
            # Test connection
            local_client.get_collections()
            clients.append(("Local", local_client))
            logger.info("✓ Qdrant Local connected (backup target)")
        except Exception as e:
            logger.warning(f"Qdrant Local not available: {e}")
    
    if not clients:
        raise RuntimeError("No Qdrant targets available. Configure QDRANT_URL or QDRANT_HOST.")
    
    logger.info(f"ETL will write to {len(clients)} target(s): {[name for name, _ in clients]}")
    
    # Setup collection on all targets
    for name, client in clients:
        logger.info(f"Setting up collection on {name}...")
        setup_qdrant_collection(client, qdrant_config)
    
    # Fetch shareholder data if enabled
    shareholders_map: Dict[str, List[Dict[str, Any]]] = {}
    if etl_config.include_shareholders:
        logger.info("Fetching shareholder data...")
        shareholders_map = fetch_all_shareholders(mysql_config)
        logger.info(f"Loaded shareholders for {len(shareholders_map)} companies")
    
    # Fetch dividend data
    logger.info("Fetching dividend data...")
    dividends_map = fetch_all_dividends(mysql_config)
    logger.info(f"Loaded dividends for {len(dividends_map)} companies")
    
    # Fetch financial statement data (ks_financial + ks_financial_bank)
    logger.info("Fetching financial statement data...")
    financials_map = fetch_all_financials(mysql_config)
    logger.info(f"Loaded financials for {len(financials_map)} companies")
    
    # Get row count for progress bar
    try:
        total_rows = count_rows(mysql_config)
        if etl_config.max_rows:
            total_rows = min(total_rows, etl_config.max_rows)
        logger.info(f"Total rows to process: {total_rows:,}")
    except Exception as e:
        logger.warning(f"Could not get row count: {e}")
        total_rows = None
    
    # Process in batches
    batch: List[Dict[str, Any]] = []
    total_processed = 0
    batch_count = 0
    
    with tqdm(total=total_rows, desc="Processing rows", unit="rows") as pbar:
        for row in stream_rows(mysql_config, etl_config.max_rows):
            batch.append(row)
            
            # Process batch when full
            if len(batch) >= etl_config.batch_size:
                try:
                    # Upsert to ALL targets
                    for target_name, client in clients:
                        count = process_and_upsert_batch(
                            batch=batch,
                            embedder=embedder,
                            client=client,
                            config=qdrant_config,
                            start_id=total_processed + 1,
                            shareholders_map=shareholders_map,
                            dividends_map=dividends_map,
                            financials_map=financials_map,
                            dry_run=etl_config.dry_run,
                        )
                    total_processed += len(batch)
                    batch_count += 1
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_count}: {e}")
                    # Continue with next batch instead of failing completely
                    
                batch = []
        
        # Process remaining rows
        if batch:
            try:
                # Upsert to ALL targets
                for target_name, client in clients:
                    count = process_and_upsert_batch(
                        batch=batch,
                        embedder=embedder,
                        client=client,
                        config=qdrant_config,
                        start_id=total_processed + 1,
                        shareholders_map=shareholders_map,
                        dividends_map=dividends_map,
                        financials_map=financials_map,
                        dry_run=etl_config.dry_run,
                    )
                total_processed += len(batch)
                batch_count += 1
                pbar.update(len(batch))
                
            except Exception as e:
                logger.error(f"Error processing final batch: {e}")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    stats = {
        "total_rows_processed": total_processed,
        "total_batches": batch_count,
        "batch_size": etl_config.batch_size,
        "duration_seconds": duration,
        "rows_per_second": total_processed / duration if duration > 0 else 0,
        "collection_name": qdrant_config.collection_name,
        "shareholders_loaded": len(shareholders_map),
        "dividends_loaded": len(dividends_map),
        "financials_loaded": len(financials_map),
        "dry_run": etl_config.dry_run,
    }
    
    logger.info("=" * 60)
    logger.info("ETL Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Total rows processed: {stats['total_rows_processed']:,}")
    logger.info(f"Total batches: {stats['total_batches']:,}")
    logger.info(f"Shareholders loaded for: {stats['shareholders_loaded']} companies")
    logger.info(f"Financials loaded for: {stats['financials_loaded']} companies")
    logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")
    logger.info(f"Throughput: {stats['rows_per_second']:.2f} rows/second")
    logger.info(f"Collection: {stats['collection_name']}")
    
    return stats


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for the ETL pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MySQL to Qdrant ETL Pipeline with BGE-M3 Hybrid Search'
    )
    
    # MySQL options
    parser.add_argument('--mysql-host', default=os.getenv('MYSQL_HOST', 'localhost'))
    parser.add_argument('--mysql-port', type=int, default=int(os.getenv('MYSQL_PORT', 3306)))
    parser.add_argument('--mysql-user', default=os.getenv('MYSQL_USER', 'root'))
    parser.add_argument('--mysql-password', default=os.getenv('MYSQL_PASSWORD', 'root'))
    parser.add_argument('--mysql-database', default=os.getenv('MYSQL_DATABASE', 'idnrag'))
    parser.add_argument('--mysql-table', default=os.getenv('MYSQL_TABLE', 'company'))
    
    # Qdrant options
    parser.add_argument('--qdrant-host', default=os.getenv('QDRANT_HOST', 'localhost'))
    parser.add_argument('--qdrant-port', type=int, default=int(os.getenv('QDRANT_PORT', 6333)))
    parser.add_argument('--collection', default=os.getenv('QDRANT_COLLECTION', 'financial_data'))
    
    # ETL options
    parser.add_argument('--batch-size', type=int, default=50, help='Rows per batch')
    parser.add_argument('--max-rows', type=int, default=None, help='Limit rows to process')
    parser.add_argument('--dry-run', action='store_true', help='Skip actual upserts')
    parser.add_argument('--no-shareholders', action='store_true', help='Skip shareholder data')
    
    args = parser.parse_args()
    
    # Build configs
    mysql_config = MySQLConfig(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        database=args.mysql_database,
        table=args.mysql_table,
    )
    
    qdrant_config = QdrantConfig(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection_name=args.collection,
    )
    
    etl_config = ETLConfig(
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        dry_run=args.dry_run,
        include_shareholders=not args.no_shareholders,
    )
    
    # Run ETL
    try:
        stats = run_etl(mysql_config, qdrant_config, etl_config)
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("ETL interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ETL failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
