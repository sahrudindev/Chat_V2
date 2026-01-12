#!/usr/bin/env python3
"""
System Diagnostic Script
========================
Checks all connections and dependencies for the ETL pipeline.
"""

import sys

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_result(name, success, details=""):
    status = "✅" if success else "❌"
    print(f"{status} {name}")
    if details:
        print(f"   └─ {details}")

# ============================================================
# 1. CHECK PYTHON VERSION
# ============================================================
print_header("1. Python Environment")
check_result(
    f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    sys.version_info >= (3, 10),
    sys.executable
)

# ============================================================
# 2. CHECK REQUIRED PACKAGES
# ============================================================
print_header("2. Required Packages")

packages = [
    ('pymysql', 'pymysql'),
    ('qdrant_client', 'qdrant-client'),
    ('FlagEmbedding', 'FlagEmbedding'),
    ('torch', 'PyTorch'),
    ('transformers', 'transformers'),
    ('tqdm', 'tqdm'),
]

for module, name in packages:
    try:
        __import__(module)
        check_result(name, True)
    except ImportError as e:
        check_result(name, False, str(e))

# ============================================================
# 3. CHECK GPU/CUDA
# ============================================================
print_header("3. GPU / Compute Device")

try:
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        check_result(f"GPU: {device_name}", True, f"VRAM: {vram:.1f} GB")
        
        # Check ROCm vs CUDA
        if hasattr(torch.version, 'hip') and torch.version.hip:
            check_result("ROCm/HIP Backend", True, f"Version: {torch.version.hip}")
        else:
            check_result("CUDA Backend", True, f"Version: {torch.version.cuda}")
    else:
        check_result("GPU", False, "No GPU detected, will use CPU")
except Exception as e:
    check_result("GPU Check", False, str(e))

# ============================================================
# 4. CHECK MYSQL CONNECTION
# ============================================================
print_header("4. MySQL Connection")

try:
    import pymysql
    import os
    
    config = {
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'port': int(os.getenv('MYSQL_PORT', 3306)),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', 'root'),
        'database': os.getenv('MYSQL_DATABASE', 'local_news'),
    }
    
    conn = pymysql.connect(**config)
    check_result(f"MySQL {config['host']}:{config['port']}", True, f"Database: {config['database']}")
    
    # Get table info
    table = os.getenv('MYSQL_TABLE', 'stock_session1_data')
    with conn.cursor() as cursor:
        # Check if table exists and get columns
        cursor.execute(f"DESCRIBE {table}")
        columns = [row[0] for row in cursor.fetchall()]
        check_result(f"Table: {table}", True, f"Columns: {', '.join(columns)}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        check_result(f"Row Count", True, f"{count:,} rows")
        
        # Sample data
        cursor.execute(f"SELECT * FROM {table} LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            print(f"\n   📋 Sample Row:")
            for col, val in zip(columns, sample):
                print(f"      {col}: {val}")
    
    conn.close()
    
except Exception as e:
    check_result("MySQL", False, str(e))

# ============================================================
# 5. CHECK QDRANT CONNECTION
# ============================================================
print_header("5. Qdrant Connection")

try:
    from qdrant_client import QdrantClient
    import os
    
    host = os.getenv('QDRANT_HOST', 'localhost')
    port = int(os.getenv('QDRANT_PORT', 6333))
    
    client = QdrantClient(host=host, port=port, timeout=5)
    collections = client.get_collections().collections
    
    check_result(f"Qdrant {host}:{port}", True, f"{len(collections)} collections")
    
    if collections:
        print(f"\n   📦 Existing Collections:")
        for coll in collections:
            print(f"      - {coll.name}")
    
except Exception as e:
    check_result("Qdrant", False, str(e))

# ============================================================
# 6. CHECK BGE-M3 MODEL (Quick Load Test)
# ============================================================
print_header("6. BGE-M3 Model (Quick Test)")

try:
    print("   ⏳ Loading model (this may take a moment)...")
    from FlagEmbedding import BGEM3FlagModel
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=(device=='cuda'), device=device)
    
    # Quick test
    test_text = "BBCA reported Net Profit of 45 billion"
    output = model.encode(test_text, return_dense=True, return_sparse=True)
    
    dense_dim = len(output['dense_vecs'])
    sparse_tokens = len(output['lexical_weights'])
    
    check_result("BGE-M3 Model Loaded", True, f"Device: {device}")
    check_result("Dense Vector", True, f"Dimension: {dense_dim}")
    check_result("Sparse Vector", True, f"Tokens: {sparse_tokens}")
    
except Exception as e:
    check_result("BGE-M3 Model", False, str(e))

# ============================================================
# SUMMARY
# ============================================================
print_header("DIAGNOSTIC COMPLETE")
print("\nRun the ETL with the correct columns from your MySQL table.")
print("Example: python mysql_to_qdrant_etl.py --dry-run --max-rows 10\n")
