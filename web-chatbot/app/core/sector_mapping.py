"""
IDX Industrial Classification (IDX-IC) Sector Mapping

Format: Sektor (1 huruf) → Sub-sektor → Industri → Sub-industri
"""

# Sektor (Level 1)
SECTORS = {
    "A": "Energi",
    "B": "Bahan Baku",
    "C": "Perindustrian",
    "D": "Barang Konsumen Primer",
    "E": "Barang Konsumen Non-Primer",
    "F": "Kesehatan",
    "G": "Keuangan",
    "H": "Properti & Real Estat",
    "I": "Teknologi",
    "J": "Infrastruktur",
    "K": "Transportasi & Logistik",
    "Z": "Pengembangan",
}

# Sub-industri (Level 4 - kode lengkap 4 karakter)
SUB_INDUSTRIES = {
    # Energi
    "A111": "Minyak & Gas Bumi",
    "A112": "Batubara",
    
    # Bahan Baku
    "B111": "Logam & Mineral Tambang",
    "B112": "Bahan Kimia",
    "B121": "Wadah & Kemasan",
    "B131": "Kertas & Produk Kertas",
    
    # Perindustrian
    "C111": "Mesin & Alat Berat",
    "C121": "Konstruksi Bangunan",
    "C131": "Jasa Komersial & Profesional",
    
    # Barang Konsumen Primer
    "D111": "Makanan & Minuman",
    "D112": "Tembakau",
    "D121": "Farmasi",
    "D131": "Produk Kebutuhan Rumah Tangga",
    "D141": "Peritel Makanan & Kebutuhan Pokok",
    
    # Barang Konsumen Non-Primer
    "E111": "Produk Rumah Tangga & Kantor",
    "E112": "Tekstil & Produk Tekstil",
    "E113": "Alas Kaki",
    "E121": "Otomotif & Komponen",
    "E131": "Media & Hiburan",
    "E141": "Peritel Non-Makanan",
    "E151": "Waktu Luang & Produk Konsumen",
    "E161": "Produk Konsumen Hotel, Restoran & Pariwisata",
    
    # Kesehatan
    "F111": "Penyedia Layanan Kesehatan",
    "F121": "Industri Farmasi",
    
    # Keuangan
    "G111": "Bank",
    "G112": "Lembaga Pembiayaan",
    "G121": "Asuransi",
    "G131": "Sekuritas & Investasi",
    "G141": "Jasa Keuangan Lainnya",
    
    # Properti & Real Estat
    "H111": "Properti & Real Estat",
    
    # Teknologi
    "I111": "Perangkat Keras Teknologi",
    "I112": "Perangkat Lunak & Layanan TI",
    
    # Infrastruktur - EXPANDED with 3-digit codes
    "J111": "Utilitas",
    "J121": "Telekomunikasi",
    "J131": "Jalan Tol, Pelabuhan, Bandara & Sejenisnya",
    # 3-digit variations (some companies use these)
    "J11": "Utilitas",
    "J12": "Telekomunikasi",
    "J13": "Infrastruktur Transportasi",
    "J31": "Telekomunikasi",  # Alternative code for telecom
    "J312": "Telekomunikasi",  # TLKM uses this code
    "J211": "Telekomunikasi",
    "J21": "Telekomunikasi",
    
    # Transportasi & Logistik
    "K111": "Transportasi",
    "K121": "Logistik & Pengiriman",
    "K11": "Transportasi",
    "K12": "Logistik",
    
    # Pengembangan
    "Z": "Papan Pengembangan",
}


def get_sector_name(code: str) -> str:
    """
    Convert sector code to readable name.
    
    Args:
        code: Sector code like "H111", "G111", "A"
        
    Returns:
        Readable sector name with code, e.g., "Properti & Real Estat (H111)"
    """
    if not code:
        return "N/A"
    
    code = code.strip().upper()
    
    # Try full sub-industry code first (4 chars)
    if code in SUB_INDUSTRIES:
        return f"{SUB_INDUSTRIES[code]} ({code})"
    
    # Try sector letter only (1 char)
    if len(code) >= 1:
        sector_letter = code[0]
        if sector_letter in SECTORS:
            return f"{SECTORS[sector_letter]} ({code})"
    
    # Return original code if no mapping found
    return code


def get_sector_letter(code: str) -> str:
    """Get sector letter from any code."""
    if not code:
        return ""
    return code[0].upper() if code else ""
