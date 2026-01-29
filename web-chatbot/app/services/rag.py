"""
RAG (Retrieval-Augmented Generation) Service.
Uses remote AI service for embedding and chat.
"""
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from app.core.qdrant import QdrantService
from app.core.ai_client import AIServiceClient
from app.core.query_parser import query_parser
from app.core.sector_mapping import get_sector_name
from app.schemas import SourceDocument

logger = logging.getLogger(__name__)

# Load system prompt
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "system3.txt").read_text()


class RAGService:
    """RAG service using remote AI service for embedding and chat."""
    
    def __init__(
        self,
        ai_client: AIServiceClient,
        qdrant_service: QdrantService,
        top_k: int = 10,  # Increased for better coverage on broad queries
        score_threshold: float = 0.3
    ):
        self.ai_client = ai_client
        self.qdrant_service = qdrant_service
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    async def retrieve(self, query: str) -> Tuple[List[SourceDocument], str]:
        """
        Retrieve relevant documents for a query with hybrid search.
        """
        # Parse query for filters
        parsed = query_parser.parse(query)
        search_text = parsed.search_text
        qdrant_filter = query_parser.filters_to_qdrant(parsed.filters)
        
        # Use parsed values for sorting and count
        sort_field = parsed.sort_field
        sort_descending = parsed.sort_descending
        requested_count = parsed.requested_count
        
        # Fallback sort field from filter if not determined
        if not sort_field and parsed.filters:
            for f in parsed.filters:
                sort_field = f.get("key")
                break
        
        logger.info(f"Query: count={requested_count}, sort={sort_field}, desc={sort_descending}, filters={parsed.filters}")
        
        if parsed.filters or sort_field:
            # Use PURE FILTER/SCROLL for filter queries OR ranking queries (with sort_field)
            # This ensures ranking queries get ALL data for accurate sorting
            logger.info(f"Using scroll_with_filter: filter={qdrant_filter}")
            results = self.qdrant_service.scroll_with_filter(
                query_filter=qdrant_filter,
                limit=1000  # Get ALL data for accurate ranking (BEI has ~800 stocks)
            )
            # If explicit filter was used but few results (maybe checking semantic relevance too),
            # we might want to still do hybrid search? No, filter is authoritative.
            
            logger.info(f"Scroll returned {len(results)} results")
            
            # Sort results by sort field if specified
            if sort_field and results:
                results = sorted(
                    results,
                    key=lambda x: x.get("payload", {}).get(sort_field) or 0,
                    reverse=sort_descending
                )
                logger.info(f"Sorted by {sort_field}, first 3: {[r.get('payload',{}).get('exchange') for r in results[:3]]}")
                
                # DUAL-CRITERIA DETECTION: Check if query asks for both gainers AND losers
                query_lower = query.lower()
                is_dual_criteria = (
                    ('dan' in query_lower or 'and' in query_lower) and
                    any(kw in query_lower for kw in ['keuntungan', 'gainer', 'naik', 'untung']) and
                    any(kw in query_lower for kw in ['kerugian', 'loser', 'turun', 'rugi'])
                )
                
                if is_dual_criteria and sort_field == 'percentage_price':
                    # For dual-criteria, we need BOTH top gainers AND top losers
                    # Current results are sorted one way, get the opposite too
                    logger.info("Dual-criteria detected: getting both gainers and losers")
                    
                    # Top 10 from current sort (gainers if desc, losers if asc)
                    first_set = results[:10]
                    
                    # Sort opposite direction for the other criteria
                    opposite_results = sorted(
                        results,
                        key=lambda x: x.get("payload", {}).get(sort_field) or 0,
                        reverse=not sort_descending  # Opposite direction
                    )
                    second_set = opposite_results[:10]
                    
                    # Combine both sets, removing duplicates
                    seen_ids = {r.get("id") for r in first_set}
                    for r in second_set:
                        if r.get("id") not in seen_ids:
                            first_set.append(r)
                            seen_ids.add(r.get("id"))
                    
                    results = first_set
                    logger.info(f"Dual-criteria: combined {len(results)} results")
        else:
            # Use semantic vector search ONLY for descriptive queries (no filter, no sort)
            dense_vector, sparse_vector = await self.ai_client.embed(search_text)
            results = self.qdrant_service.search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=1000,  # Full dataset coverage (~800 stocks)
                score_threshold=self.score_threshold,
                query_filter=None
            )
        
        # SHAREHOLDER CROSS-REFERENCE: Detect if query is asking about a person at a specific company
        # Pattern: "Apakah [NAME] pemegang saham [COMPANY]?"
        query_lower = query.lower()
        shareholder_keywords = ['pemegang saham', 'shareholder', 'kepemilikan', 'dimiliki']
        is_shareholder_query = any(kw in query_lower for kw in shareholder_keywords)
        
        if is_shareholder_query and results:
            # Check if the top result actually contains info about the queried entity
            # If not, search semantically for the person name
            top_result = results[0] if results else {}
            top_text = top_result.get("payload", {}).get("text", "").lower()
            
            # Try to extract person/entity names from query (words before shareholder keywords)
            import re
            # Common patterns: "Apakah X pemegang saham Y", "X adalah pemegang saham Y"
            name_match = re.search(r'(?:apakah|adalah)\s+([^?]+?)\s+(?:pemegang|shareholder)', query_lower)
            if name_match:
                entity_name = name_match.group(1).strip()
                # Check if entity exists in top results
                entity_found = any(entity_name in r.get("payload", {}).get("text", "").lower() for r in results[:5])
                
                if not entity_found and len(entity_name) > 2:
                    # Do secondary search for the person/entity name
                    logger.info(f"Shareholder cross-ref: searching for entity '{entity_name}'")
                    entity_dense, entity_sparse = await self.ai_client.embed(entity_name)
                    entity_results = self.qdrant_service.search(
                        dense_vector=entity_dense,
                        sparse_vector=entity_sparse,
                        top_k=10,
                        score_threshold=self.score_threshold,
                        query_filter=None
                    )
                    # Merge entity results with original (avoiding duplicates)
                    seen_ids = {r.get("id") for r in results[:10]}
                    for er in entity_results:
                        if er.get("id") not in seen_ids:
                            results.insert(5, er)  # Insert after top 5
                            seen_ids.add(er.get("id"))
                    logger.info(f"Added {len(entity_results)} cross-reference results")
        
        # Format results - 5 detailed + rest as codes only
        sources = []
        detailed_parts = []
        additional_codes = []
        
        # Safe number formatting
        def fmt_num(val, prefix="", suffix=""):
            if val is None:
                return "N/A"
            try:
                return f"{prefix}{val:,.0f}{suffix}"
            except:
                return str(val)
        
        def fmt_pct(val):
            if val is None:
                return "N/A"
            try:
                return f"{val:.2f}%"
            except:
                return str(val)
        
        for i, result in enumerate(results):
            payload = result.get("payload", {})
            exchange = payload.get("exchange") or "N/A"
            name = payload.get("name") or "Unknown"
            
            source = SourceDocument(
                exchange=exchange,
                name=name,
                text=payload.get("text", "")[:500],
                score=result.get("score", 0.0)
            )
            sources.append(source)
            
            if i < 10:
                # First 10: Full detailed context (increased from 5 for ranking queries)
                sector_code = payload.get('si_code') or ''
                sector_display = get_sector_name(sector_code) if sector_code else 'N/A'
                
                # Format shareholders for display
                shareholders_list = payload.get('shareholders', [])
                shareholders_text = ""
                if shareholders_list:
                    shareholders_text = "\nPemegang Saham:\n"
                    for j, sh in enumerate(shareholders_list[:5]):  # Top 5 shareholders
                        sh_name = sh.get('name', 'Unknown')
                        sh_pct = sh.get('percentage', 0)
                        sh_shares = sh.get('shares', 0)
                        shareholders_text += f"  {j+1}. {sh_name}: {sh_pct:.2f}% ({sh_shares:,} saham)\n"
                    if len(shareholders_list) > 5:
                        shareholders_text += f"  ... dan {len(shareholders_list) - 5} pemegang saham lainnya\n"
                
                # Format dividends for display
                # Format dividends for display
                dividends_list = payload.get('dividends', [])
                dividends_text = ""
                if dividends_list:
                    dividends_text = "\nRiwayat Dividen:\n"
                    
                    # Group by year to handle Interim + Final
                    divs_by_year = {}
                    for div in dividends_list:
                        year = div.get('year')
                        if not year: continue
                        if year not in divs_by_year:
                             divs_by_year[year] = []
                        divs_by_year[year].append(div)
                    
                    # Sort years descending
                    sorted_years = sorted(divs_by_year.keys(), reverse=True)
                    
                    display_count = 0
                    for year in sorted_years:
                        if display_count >= 3: break
                        
                        year_divs = divs_by_year[year]
                        
                        # Calculate total for the year
                        total_dps = sum(d.get('dividend_per_share', 0) for d in year_divs)
                        total_yield = sum(d.get('dividend_yield', 0) for d in year_divs)
                        
                        # Format types (e.g., "Interim & Final")
                        types = [d.get('type', '') for d in year_divs if d.get('type')]
                        type_str = " & ".join(types) if types else ""
                        
                        # Skip future years with 0 yield if we have valid history, UNLESS it's the only data
                        if total_yield == 0 and display_count > 0:
                            continue

                        line = f"  - Tahun {year}: Total DPS Rp {total_dps:,.0f} (Total Yield: {total_yield:.2f}%)"
                        if type_str:
                            line += f" [{type_str}]"
                        
                        # Add individual details if multiple dividends in year
                        if len(year_divs) > 1:
                            details = []
                            for d in year_divs:
                                d_type = d.get('type', 'Dividend')
                                d_yield = d.get('dividend_yield', 0)
                                details.append(f"{d_type}: {d_yield:.2f}%")
                            line += f" ({', '.join(details)})"
                            
                        dividends_text += line + "\n"
                        display_count += 1
                        
                    if len(sorted_years) > 3:
                        dividends_text += f"  ... dan riwayat tahun-tahun sebelumnya\n"
                
                # Format financial data for display
                financial_text = ""
                if payload.get('financial_period'):
                    is_bank = payload.get('is_bank', False)
                    period = payload.get('financial_period', 'N/A')
                    
                    # Common metrics
                    financial_text = f"\n📊 Laporan Keuangan ({period}):\n"
                    
                    net_profit = payload.get('net_profit')
                    if net_profit:
                        yoy = payload.get('net_profit_yoy_growth')
                        yoy_str = f" ({yoy:+.1f}% YoY)" if yoy else ""
                        financial_text += f"  - Net Profit: {fmt_num(net_profit, 'Rp ')}{yoy_str}\n"
                    
                    roe = payload.get('roe')
                    roa = payload.get('roa')
                    if roe:
                        financial_text += f"  - ROE: {roe:.2f}%\n"
                    if roa:
                        financial_text += f"  - ROA: {roa:.2f}%\n"
                    
                    # EPS from financial
                    eps = payload.get('eps') or payload.get('earning_per_share')
                    if eps:
                        financial_text += f"  - EPS: {eps:,.2f}\n"
                    
                    if not is_bank:
                        # Non-bank metrics
                        revenue = payload.get('revenue')
                        if revenue:
                            rev_yoy = payload.get('revenue_yoy_growth')
                            rev_yoy_str = f" ({rev_yoy:+.1f}% YoY)" if rev_yoy else ""
                            financial_text += f"  - Revenue: {fmt_num(revenue, 'Rp ')}{rev_yoy_str}\n"
                        
                        cogs = payload.get('cogs')
                        if cogs:
                            financial_text += f"  - COGS: {fmt_num(cogs, 'Rp ')}\n"
                        
                        gross_profit = payload.get('gross_profit')
                        if gross_profit:
                            financial_text += f"  - Gross Profit: {fmt_num(gross_profit, 'Rp ')}\n"
                        
                        operating_profit = payload.get('operating_profit')
                        if operating_profit:
                            financial_text += f"  - Operating Profit: {fmt_num(operating_profit, 'Rp ')}\n"
                        
                        gross_margin = payload.get('gross_margin')
                        operating_margin = payload.get('operating_margin')
                        net_margin = payload.get('net_margin')
                        if gross_margin:
                            financial_text += f"  - Gross Margin: {gross_margin:.1f}%\n"
                        if operating_margin:
                            financial_text += f"  - Operating Margin: {operating_margin:.1f}%\n"
                        if net_margin:
                            financial_text += f"  - Net Margin: {net_margin:.1f}%\n"
                    else:
                        # Bank metrics
                        nii = payload.get('net_interest_income')
                        if nii:
                            financial_text += f"  - Net Interest Income: {fmt_num(nii, 'Rp ')}\n"
                    
                    # Balance Sheet
                    total_assets = payload.get('total_assets')
                    total_liabilities = payload.get('total_liabilities')
                    total_equity = payload.get('total_equity')
                    if total_assets:
                        financial_text += f"  - Total Assets: {fmt_num(total_assets, 'Rp ')}\n"
                    if total_liabilities:
                        financial_text += f"  - Total Liabilities: {fmt_num(total_liabilities, 'Rp ')}\n"
                    if total_equity:
                        financial_text += f"  - Total Equity: {fmt_num(total_equity, 'Rp ')}\n"
                    
                    # Common ratios
                    pbv = payload.get('pbv_ratio')
                    der = payload.get('der_ratio')
                    if pbv:
                        financial_text += f"  - P/BV Ratio: {pbv:.2f}x\n"
                    if der:
                        financial_text += f"  - DER Ratio: {der:.2f}x\n"
                    
                    # Financial History (for trend queries) - Complete quarterly data
                    history = payload.get('financial_history', [])
                    if len(history) >= 2:
                        is_bank_company = payload.get('is_bank', False)
                        financial_text += f"\n  📈 Trend {len(history)} Kuartal:\n"
                        
                        for hist in history[:4]:
                            hist_period = hist.get('period', '')
                            financial_text += f"    [{hist_period}]\n"
                            
                            # Bank-specific - Income Statement
                            if is_bank_company:
                                if hist.get('net_interest_income'):
                                    financial_text += f"      Pendapatan Bunga Bersih: {fmt_num(hist.get('net_interest_income'), 'Rp ')}\n"
                                if hist.get('operating_expense'):
                                    financial_text += f"      Biaya Operasional: {fmt_num(hist.get('operating_expense'), 'Rp ')}\n"
                                if hist.get('operating_profit'):
                                    financial_text += f"      Laba Operasional: {fmt_num(hist.get('operating_profit'), 'Rp ')}\n"
                                if hist.get('net_profit'):
                                    financial_text += f"      Keuntungan Bersih: {fmt_num(hist.get('net_profit'), 'Rp ')}\n"
                                if hist.get('total_shares'):
                                    financial_text += f"      Total Saham: {hist.get('total_shares'):,.0f}\n"
                                if hist.get('eps'):
                                    financial_text += f"      EPS: {hist.get('eps'):,.0f}\n"
                                # Balance Sheet
                                if hist.get('total_assets'):
                                    financial_text += f"      Total Aset: {fmt_num(hist.get('total_assets'), 'Rp ')}\n"
                                if hist.get('total_liabilities'):
                                    financial_text += f"      Total Liabilitas: {fmt_num(hist.get('total_liabilities'), 'Rp ')}\n"
                                if hist.get('minority_interest'):
                                    financial_text += f"      Kepentingan Non-pengendali: {fmt_num(hist.get('minority_interest'), 'Rp ')}\n"
                                if hist.get('total_equity'):
                                    financial_text += f"      Total Ekuitas: {fmt_num(hist.get('total_equity'), 'Rp ')}\n"
                                # Ratios
                                if hist.get('roe'):
                                    financial_text += f"      ROE: {hist.get('roe'):.2f}%\n"
                                if hist.get('roa'):
                                    financial_text += f"      ROA: {hist.get('roa'):.2f}%\n"
                                # Cash Flow
                                if hist.get('net_cash'):
                                    financial_text += f"      Kas Bersih Operasional: {fmt_num(hist.get('net_cash'), 'Rp ')}\n"
                                if hist.get('cash_on_hand'):
                                    financial_text += f"      Dana Tunai Akhir Periode: {fmt_num(hist.get('cash_on_hand'), 'Rp ')}\n"
                            else:
                                # Non-bank specific
                                if hist.get('revenue'):
                                    financial_text += f"      Revenue: {fmt_num(hist.get('revenue'), 'Rp ')}\n"
                                if hist.get('cogs'):
                                    financial_text += f"      COGS: {fmt_num(hist.get('cogs'), 'Rp ')}\n"
                                if hist.get('gross_profit'):
                                    financial_text += f"      Gross Profit: {fmt_num(hist.get('gross_profit'), 'Rp ')}\n"
                                if hist.get('operating_profit'):
                                    financial_text += f"      Operating Profit: {fmt_num(hist.get('operating_profit'), 'Rp ')}\n"
                                if hist.get('net_profit'):
                                    financial_text += f"      Net Profit: {fmt_num(hist.get('net_profit'), 'Rp ')}\n"
                                if hist.get('eps'):
                                    financial_text += f"      EPS: {hist.get('eps'):,.0f}\n"
                                if hist.get('gross_margin'):
                                    financial_text += f"      Gross Margin: {hist.get('gross_margin'):.1f}%\n"
                                if hist.get('operating_margin'):
                                    financial_text += f"      Operating Margin: {hist.get('operating_margin'):.1f}%\n"
                                if hist.get('net_margin'):
                                    financial_text += f"      Net Margin: {hist.get('net_margin'):.1f}%\n"
                                # Balance sheet
                                if hist.get('total_assets'):
                                    financial_text += f"      Total Assets: {fmt_num(hist.get('total_assets'), 'Rp ')}\n"
                                if hist.get('total_liabilities'):
                                    financial_text += f"      Total Liabilities: {fmt_num(hist.get('total_liabilities'), 'Rp ')}\n"
                                if hist.get('total_equity'):
                                    financial_text += f"      Total Equity: {fmt_num(hist.get('total_equity'), 'Rp ')}\n"
                                # Ratios
                                if hist.get('roe'):
                                    financial_text += f"      ROE: {hist.get('roe'):.2f}%\n"
                                if hist.get('roa'):
                                    financial_text += f"      ROA: {hist.get('roa'):.2f}%\n"
                
                detailed_parts.append(f"""
[Perusahaan {i+1}]
Nama: {name} ({exchange})
Sektor: {sector_display}
Tipe: {"Bank" if payload.get('is_bank') else "Non-Bank"}
Established: {payload.get('established') or 'N/A'}
Listing Date: {payload.get('listing') or 'N/A'}
Harga Penutupan: {fmt_num(payload.get('close_price'), "Rp ")}
Perubahan Harga: {fmt_num(payload.get('price_change'))} ({fmt_pct(payload.get('percentage_price'))})
Kapitalisasi Pasar: {fmt_num(payload.get('capitalization'), "Rp ")}
P/E Ratio: {payload.get('price_earning_ratio') or 'N/A'}
EPS: {payload.get('earning_per_share') or 'N/A'}{financial_text}{shareholders_text}{dividends_text}
""")
            else:
                # Rest: Just code
                additional_codes.append(exchange)
        
        # Build context
        context = "\n".join(detailed_parts)
        
        # Add additional codes if any
        if additional_codes:
            codes_str = ", ".join(additional_codes[:10])  # Max 10 codes
            remaining = len(additional_codes) - 10 if len(additional_codes) > 10 else 0
            if remaining > 0:
                context += f"\n\n[Data Tambahan]\nSelain 10 perusahaan di atas, ada juga: {codes_str}, dan {remaining} perusahaan lainnya."
            else:
                context += f"\n\n[Data Tambahan]\nSelain 10 perusahaan di atas, ada juga: {codes_str}."
            context += "\nSarankan user untuk cek idnfinancials.com untuk informasi lengkap."
        
        # Add DIVIDEND RANKING SUMMARY for all results (for dividend ranking queries)
        # This provides compact dividend data for ALL companies to enable proper ranking
        dividend_summary_parts = []
        for result in results:
            payload = result.get("payload", {})
            div_yield = payload.get("latest_dividend_yield", 0)
            div_dps = payload.get("latest_dividend_per_share", 0)
            div_year = payload.get("latest_dividend_year")
            if div_yield and div_yield > 0:
                name = payload.get("name", "Unknown")
                exchange = payload.get("exchange", "N/A")
                dividend_summary_parts.append(
                    f"- {name} ({exchange}): Tahun {div_year}, DPS Rp {div_dps:,.0f}, Yield {div_yield:.2f}%"
                )
        
        if dividend_summary_parts:
            context += "\n\n[RINGKASAN DIVIDEN SEMUA PERUSAHAAN]\n"
            context += "Berikut data dividend yield dari semua perusahaan yang tersedia (untuk ranking):\n"
            context += "\n".join(dividend_summary_parts[:50])  # Max 50 for token limit
            if len(dividend_summary_parts) > 50:
                context += f"\n...dan {len(dividend_summary_parts) - 50} perusahaan lainnya"
        
        # Add GAINERS + LOSERS SUMMARY for dual-criteria queries
        query_lower = query.lower()
        is_dual_criteria = (
            ('dan' in query_lower or 'and' in query_lower) and
            any(kw in query_lower for kw in ['keuntungan', 'gainer', 'naik', 'untung']) and
            any(kw in query_lower for kw in ['kerugian', 'loser', 'turun', 'rugi'])
        )
        
        if is_dual_criteria:
            # Sort all results by percentage_price for gainers/losers summary
            sorted_by_price = sorted(
                results,
                key=lambda x: x.get("payload", {}).get("percentage_price") or 0,
                reverse=True
            )
            
            # Top 5 Gainers (highest % change)
            gainers_parts = ["\n\n[TOP GAINERS - Keuntungan Terbesar]"]
            for r in sorted_by_price[:5]:
                p = r.get("payload", {})
                gainers_parts.append(f"- {p.get('name')} ({p.get('exchange')}): {p.get('percentage_price', 0):+.2f}%")
            context += "\n".join(gainers_parts)
            
            # Top 5 Losers (lowest/most negative % change)
            losers_parts = ["\n\n[TOP LOSERS - Kerugian Terbesar]"]
            for r in sorted_by_price[-5:][::-1]:  # Last 5, reversed to show most negative first
                p = r.get("payload", {})
                pct = p.get('percentage_price', 0)
                if pct < 0:  # Only show actual losers
                    losers_parts.append(f"- {p.get('name')} ({p.get('exchange')}): {pct:+.2f}%")
            if len(losers_parts) > 1:
                context += "\n".join(losers_parts)
            else:
                context += "\n\n[TOP LOSERS - Kerugian Terbesar]\nTidak ada saham dengan perubahan negatif dalam data saat ini."
        
        # === BANK VS NON-BANK COMPARISON VALIDATION ===
        # Detect financial comparison queries and validate same-type comparison
        comparison_keywords = ['bandingkan', 'perbandingan', 'perbedaan', 'vs', 'versus', 'dibanding']
        financial_keywords = ['keuangan', 'laporan', 'finansial', 'roe', 'roa', 'profit', 'revenue', 'laba']
        
        is_comparison = any(kw in query_lower for kw in comparison_keywords)
        is_financial_comparison = is_comparison and any(kw in query_lower for kw in financial_keywords)
        
        if is_financial_comparison and len(results) >= 2:
            # Check company types from first 2 results
            companies_info = []
            for r in results[:2]:
                p = r.get("payload", {})
                is_bank = p.get("is_bank", False)
                si_code = p.get("si_code", "")
                # Also detect bank from sector code (G111 = Bank)
                is_bank_from_sector = si_code.startswith("G111") if si_code else False
                companies_info.append({
                    "name": p.get("name", "Unknown"),
                    "exchange": p.get("exchange", "N/A"),
                    "is_bank": is_bank or is_bank_from_sector
                })
            
            # Check if types are different (one bank, one non-bank)
            if len(companies_info) >= 2:
                c1, c2 = companies_info[0], companies_info[1]
                if c1["is_bank"] != c2["is_bank"]:
                    # Mixed types - inject warning into context
                    bank_name = c1["exchange"] if c1["is_bank"] else c2["exchange"]
                    non_bank_name = c2["exchange"] if c1["is_bank"] else c1["exchange"]
                    
                    warning = f"""
[⚠️ PERINGATAN: PERBANDINGAN TIDAK VALID]
User meminta perbandingan finansial antara:
- {bank_name} (BANK)
- {non_bank_name} (NON-BANK)

ATURAN: Perbandingan laporan keuangan hanya valid untuk sesama jenis:
- Bank vs Bank ✅
- Non-Bank vs Non-Bank ✅
- Bank vs Non-Bank ❌ (tidak valid karena struktur laporan berbeda)

RESPON YANG HARUS DIBERIKAN:
"Maaf, perbandingan laporan keuangan antara bank ({bank_name}) dan non-bank ({non_bank_name}) 
tidak dapat dilakukan karena struktur laporan keuangan yang berbeda. 
Silakan bandingkan sesama bank atau sesama non-bank."
"""
                    context = warning + "\n" + context
                    logger.info(f"Bank vs Non-Bank comparison detected: {bank_name} vs {non_bank_name}")
        
        return sources, context
    
    async def generate_response(
        self,
        query: str,
        context: str,
        history: Optional[List[dict]] = None
    ) -> str:
        """
        Generate response using AI service chat.
        
        Args:
            query: User query
            context: Retrieved context
            history: Chat history (optional)
            
        Returns:
            Generated response text
        """
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add history if provided (use last 8 messages for context)
        if history:
            logger.info(f"[RAG] Adding {len(history)} history messages to prompt")
            for msg in history[-8:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add context and query
        if context:
            user_message = f"""<context>
{context}
</context>

User Question: {query}"""
        else:
            user_message = f"""User Question: {query}

Note: No relevant context was found in the knowledge base."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate via AI service
        response = await self.ai_client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        return response
    
    async def chat(
        self,
        query: str,
        history: Optional[List[dict]] = None
    ) -> Tuple[str, List[SourceDocument]]:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            query: User query
            history: Chat history
            
        Returns:
            Tuple of (response, source_documents)
        """
        logger.info(f"Processing query: {query[:50]}...")
        
        # Enhance query with context from history (extract stock codes/company names)
        enhanced_query = self._enhance_query_with_history(query, history)
        if enhanced_query != query:
            logger.info(f"Enhanced query: {enhanced_query[:80]}...")
        
        # Retrieve using enhanced query
        sources, context = await self.retrieve(enhanced_query)
        logger.info(f"Retrieved {len(sources)} documents")
        
        # Generate response with original query but enhanced context
        response = await self.generate_response(query, context, history)
        logger.info("Generated response")
        
        return response, sources
    
    def _enhance_query_with_history(self, query: str, history: Optional[List[dict]]) -> str:
        """
        Enhance the current query with relevant context from history.
        
        Extracts stock codes and company names from recent history and adds
        them to the query if the current query appears to be a follow-up question.
        """
        if not history:
            return query
        
        import re
        
        # Check if this looks like a follow-up question (short, uses pronouns, etc.)
        follow_up_patterns = [
            r'^berapa\s+(harga|nilai|volume|market\s*cap|kapitalisasi)',
            r'^(apa|siapa|kapan|dimana|bagaimana)\s+(itu|saja|dia)',
            r'^(harga|nilai|volume|eps|per)nya',
            r'^(ceritakan|jelaskan)\s+(lebih|lagi)',
            r'^(dan|lalu|terus)\s+',
            r'^(bandingkan|compare)',
            r'(nya\??|itu\??|ini\??)$',  # Ends with -nya, itu, ini
        ]
        
        query_lower = query.lower().strip()
        is_follow_up = any(re.search(pattern, query_lower) for pattern in follow_up_patterns)
        
        if not is_follow_up:
            return query
        
        # Extract stock codes (4 uppercase letters) and company names from history
        stock_codes = set()
        company_keywords = []
        
        # Common stock code pattern (4 capital letters)
        stock_pattern = re.compile(r'\b([A-Z]{4})\b')
        
        for msg in history[-4:]:  # Last 4 messages
            content = msg.get("content", "")
            
            # Find stock codes
            codes = stock_pattern.findall(content)
            for code in codes:
                # Filter out common non-stock words
                if code not in ['YANG', 'AKAN', 'DARI', 'ATAU', 'PADA', 'KAMI', 'JIKA', 'BISA', 'BERI', 'TAHU', 'DATA', 'INFO']:
                    stock_codes.add(code)
            
            # Find company names mentioned after stock codes (in parentheses or after)
            name_pattern = re.compile(r'([A-Z]{4})\s*(?:\(([^)]+)\)|adalah\s+([^,.]+))')
            for match in name_pattern.finditer(content):
                if match.group(2):
                    company_keywords.append(match.group(2).strip()[:30])
                if match.group(3):
                    company_keywords.append(match.group(3).strip()[:30])
        
        # Build enhanced query
        if stock_codes:
            codes_str = " ".join(sorted(stock_codes))
            enhanced = f"{query} {codes_str}"
            logger.info(f"[RAG] Added context from history: {codes_str}")
            return enhanced
        
        return query


_rag_service = None


def get_rag_service(
    ai_client: AIServiceClient,
    qdrant_service: QdrantService,
    top_k: int = 5,
    score_threshold: float = 0.3
) -> RAGService:
    """Get singleton RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(
            ai_client,
            qdrant_service,
            top_k,
            score_threshold
        )
    return _rag_service
