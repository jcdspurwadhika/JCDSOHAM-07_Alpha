"""Customer Churn Predictor - Streamlit app (Final Project)."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "xgboost_churn_model_tuned.pkl"
META_PATH = APP_DIR / "app_meta.pkl"

# Same cost assumptions as the notebook and the Tableau dashboard
CLV = 200
RETENTION_COST = 20

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(META_PATH)


model, meta = load_artifacts()

# Defaults for numeric columns that are not in the saved medians
FALLBACK_DEFAULTS = {"NumberOfDeviceRegistered": 4, "NumberOfAddress": 3, "CashbackAmount": 164}


def default_for(col):
    return meta["medians"].get(col, FALLBACK_DEFAULTS.get(col))


def int_input(col, label, help_text=None):
    lo, hi = (int(v) for v in meta["numeric_ranges"][col])
    default = min(max(int(round(default_for(col))), lo), hi)
    return st.number_input(label, min_value=lo, max_value=hi, value=default,
                           step=1, help=help_text, key=col)


def float_input(col, label, help_text=None):
    lo, hi = (float(v) for v in meta["numeric_ranges"][col])
    default = min(max(float(default_for(col)), lo), hi)
    return st.number_input(label, min_value=lo, max_value=hi, value=default,
                           step=0.5, help=help_text, key=col)


def category_input(col, label):
    return st.selectbox(label, meta["category_options"][col], key=col)


def risk_tier(prob):
    if prob < 0.3:
        return "Rendah"
    if prob <= 0.7:
        return "Sedang"
    return "Tinggi"


# Tier statistics come from the test set (same as the Tableau dashboard)
TIER_MESSAGE = {
    "Tinggi": ("error", "Risiko tinggi: prioritaskan penawaran retensi sekarang. Pada data test, sekitar 94% pelanggan di tier ini benar-benar churn."),
    "Sedang": ("warning", "Risiko sedang: model kurang yakin (pada data test hanya sekitar 16% pelanggan di tier ini yang churn). Penawaran retensi bersifat opsional; pantau lebih dulu."),
    "Rendah": ("success", "Risiko rendah: tidak perlu intervensi. Pada data test, hanya sekitar 1% pelanggan di tier ini yang churn."),
}

TIER_ORDER = ["Tinggi", "Sedang", "Rendah"]
# Observed churn rate per tier on the held-out test set (same as the Tableau dashboard).
# Model scores are not calibrated probabilities (class weighting inflates them),
# so cost estimates use these observed rates.
TIER_CHURN_RATE = {"Tinggi": 0.9355, "Sedang": 0.1556, "Rendah": 0.0117}


def page_home():
    st.title("Customer Churn Predictor")
    st.write(
        "Aplikasi ini memprediksi kemungkinan seorang pelanggan e-commerce akan berhenti "
        "bertransaksi (*churn*), agar penawaran retensi dapat diberikan sebelum pelanggan pergi."
    )
    st.subheader("Siapa yang memakai dan untuk apa")
    st.markdown(
        "- **Pengguna utama:** tim Customer Retention/CRM.\n"
        "- **Prediksi Individu:** mengecek risiko churn satu pelanggan, misalnya saat menangani komplain.\n"
        "- **Prediksi Batch:** mengunggah daftar pelanggan dan mendapat urutan prioritas retensi.\n"
        "- **Keputusan yang dibantu:** siapa yang ditawari retensi lebih dulu saat anggaran terbatas."
    )
    st.subheader("Cara memakai")
    st.markdown(
        "1. Pilih halaman di menu kiri.\n"
        "2. Isi data pelanggan (atau unggah file), lalu jalankan prediksi.\n"
        "3. Baca probabilitas churn, *risk tier*, dan saran tindakannya."
    )
    st.warning(
        "Catatan: model dilatih pada data historis dengan label churn yang definisi pastinya "
        "tidak diketahui, sehingga paling andal untuk pelanggan dengan profil serupa data tersebut. "
        "Gunakan hasilnya sebagai alat bantu prioritas, bukan keputusan otomatis."
    )


def page_individual():
    st.title("Prediksi Individu")
    st.caption(
        "Isi data satu pelanggan lalu klik Prediksi. Nilai awal adalah nilai tipikal data latih, "
        "dan batas isian mengikuti rentang data latih."
    )

    with st.form("form_individu"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Profil pelanggan")
            tenure = int_input("Tenure", "Lama jadi pelanggan (bulan)")
            gender = category_input("Gender", "Gender")
            marital = category_input("MaritalStatus", "Status pernikahan")
            city_tier = st.selectbox(
                "City Tier", [1, 2, 3], key="CityTier",
                format_func=lambda t: {1: "Tier 1 (kota metropolitan)",
                                       2: "Tier 2 (kota menengah)", 3: "Tier 3 (kota kecil)"}[t],
            )
            warehouse = float_input("WarehouseToHome", "Jarak gudang ke rumah")

        with col2:
            st.subheader("Kebiasaan belanja")
            order_cat = category_input("PreferedOrderCat", "Kategori produk favorit")
            payment = category_input("PreferredPaymentMode", "Metode pembayaran favorit")
            login = category_input("PreferredLoginDevice", "Perangkat login favorit")
            order_count = int_input("OrderCount", "Jumlah order bulan terakhir")
            coupon = int_input("CouponUsed", "Kupon digunakan bulan terakhir")
            days = int_input("DaySinceLastOrder", "Hari sejak order terakhir")

        with col3:
            st.subheader("Interaksi dan akun")
            hike = int_input("OrderAmountHikeFromlastYear", "Kenaikan jumlah order vs tahun lalu (%)")
            cashback = int_input("CashbackAmount", "Rata-rata cashback bulan terakhir")
            hours = int_input("HourSpendOnApp", "Jam di aplikasi/website")
            devices = int_input("NumberOfDeviceRegistered", "Jumlah perangkat terdaftar")
            addresses = int_input("NumberOfAddress", "Jumlah alamat terdaftar")
            satisfaction = st.selectbox("Skor kepuasan (1-5)", [1, 2, 3, 4, 5], index=2, key="SatisfactionScore")
            complain = st.selectbox(
                "Pernah komplain?", [0, 1], key="Complain",
                format_func=lambda v: "Ya" if v == 1 else "Tidak",
            )

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        row = {
            "Tenure": tenure, "PreferredLoginDevice": login, "CityTier": city_tier,
            "WarehouseToHome": warehouse, "PreferredPaymentMode": payment, "Gender": gender,
            "HourSpendOnApp": hours, "NumberOfDeviceRegistered": devices,
            "PreferedOrderCat": order_cat, "SatisfactionScore": satisfaction,
            "MaritalStatus": marital, "NumberOfAddress": addresses, "Complain": complain,
            "OrderAmountHikeFromlastYear": hike, "CouponUsed": coupon, "OrderCount": order_count,
            "DaySinceLastOrder": days, "CashbackAmount": cashback,
        }
        input_df = pd.DataFrame([row]).reindex(columns=meta["feature_columns"])
        if input_df.isnull().any().any():
            st.error("Ada kolom fitur yang tidak cocok dengan model. Hubungi pengembang aplikasi.")
            st.stop()

        prob = float(model.predict_proba(input_df)[0, 1])
        flagged = int(model.predict(input_df)[0]) == 1
        tier = risk_tier(prob)

        st.subheader("Hasil prediksi")
        m1, m2, m3 = st.columns(3)
        m1.metric("Skor risiko churn", f"{prob:.1%}",
                  help="Skor model, bukan probabilitas terkalibrasi. Baca bersama Risk Tier.")
        m2.metric("Risk tier", tier)
        m3.metric("Ditandai model berisiko churn", "Ya" if flagged else "Tidak")

        kind, text = TIER_MESSAGE[tier]
        getattr(st, kind)(text)

        with st.expander("Lihat data yang dimasukkan"):
            st.dataframe(input_df)


def prepare_batch(raw):
    """Clean an uploaded table like the notebook did. Returns (clean features, problem notes)."""
    df = raw.copy()
    for col, mapping in meta["label_mapping"].items():
        df[col] = df[col].replace(mapping)
    for col in meta["numeric_cols"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, med in meta["medians"].items():
        df[col] = df[col].fillna(med)

    problems = pd.Series("", index=df.index)
    for col in meta["feature_columns"]:
        if col in meta["nominal_cols"]:
            invalid = ~df[col].isin(meta["category_options"][col])
        else:
            invalid = df[col].isnull()
        problems = problems.where(~invalid, problems + col + "; ")

    clean = df.loc[problems == "", meta["feature_columns"]].copy()
    clean[meta["int_cols"]] = clean[meta["int_cols"]].round().astype(int)
    return clean, problems[problems != ""]


def page_batch():
    st.title("Prediksi Batch")
    st.write("Unggah file CSV berisi banyak pelanggan untuk mendapatkan urutan prioritas retensi.")
    with st.expander("Kolom yang diperlukan"):
        st.write(", ".join(meta["feature_columns"]) + ". Kolom CustomerID bersifat opsional.")

    sample_path = APP_DIR / "contoh_pelanggan.csv"
    if sample_path.exists():
        st.download_button("Unduh contoh file CSV", sample_path.read_bytes(),
                           file_name="contoh_pelanggan.csv", mime="text/csv")

    uploaded = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception:
        st.error("File tidak dapat dibaca sebagai CSV.")
        return

    missing_cols = [c for c in meta["feature_columns"] if c not in raw.columns]
    if missing_cols:
        st.error("Kolom berikut tidak ditemukan di file: " + ", ".join(missing_cols))
        return

    clean, problems = prepare_batch(raw)
    if len(problems):
        st.warning(f"{len(problems)} baris tidak dapat diproses dan dilewati (nilai kosong atau kategori tidak dikenal).")
        with st.expander("Lihat baris yang dilewati"):
            st.dataframe(pd.DataFrame({"Baris di file": problems.index + 2,
                                       "Kolom bermasalah": problems.values}))
    if clean.empty:
        st.error("Tidak ada baris yang dapat diproses.")
        return

    outside = pd.Series(False, index=clean.index)
    for col in meta["numeric_cols"]:
        lo, hi = meta["numeric_ranges"][col]
        outside |= (clean[col] < lo) | (clean[col] > hi)
    if outside.any():
        st.info(f"{int(outside.sum())} baris memiliki nilai di luar rentang data latih; hasilnya kurang dapat diandalkan.")

    scores = model.predict_proba(clean)[:, 1]
    result = raw.loc[clean.index].drop(columns=["Skor_Risiko", "Risk_Tier"], errors="ignore")
    result.insert(0, "Risk_Tier", [risk_tier(p) for p in scores])
    result.insert(0, "Skor_Risiko", scores.round(4))
    result = result.sort_values("Skor_Risiko", ascending=False).reset_index(drop=True)

    st.subheader("Ringkasan")
    counts = result["Risk_Tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pelanggan diproses", len(result))
    c2.metric("Tier Tinggi", int(counts["Tinggi"]))
    c3.metric("Tier Sedang", int(counts["Sedang"]))
    c4.metric("Tier Rendah", int(counts["Rendah"]))

    st.subheader("Simulasi anggaran retensi")
    top_n = st.slider("Jumlah pelanggan teratas yang akan ditawari retensi",
                      0, len(result), min(int(counts["Tinggi"]), len(result)))
    top = result.head(top_n)
    cost = top_n * RETENTION_COST
    expected_churners = sum(TIER_CHURN_RATE[t] for t in top["Risk_Tier"])
    protected = expected_churners * CLV
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r1c1.metric("Biaya penawaran", f"${cost:,.0f}")
    r1c2.metric("Churner tercakup (perkiraan)", f"{expected_churners:,.0f}")
    r2c1.metric("Nilai terlindungi (perkiraan)", f"${protected:,.0f}")
    r2c2.metric("Manfaat bersih (perkiraan)", f"${protected - cost:,.0f}")
    st.caption(
        "Perkiraan memakai CLV \\$200 dan biaya retensi \\$20 per pelanggan, mengasumsikan setiap penawaran "
        "berhasil mempertahankan pelanggan, serta tingkat churn aktual tiap tier pada data test "
        "(Tinggi ~94%, Sedang ~16%, Rendah ~1%), karena skor model bukan probabilitas yang terkalibrasi."
    )

    st.subheader("Daftar prioritas")
    st.dataframe(result)
    st.download_button("Unduh hasil (CSV)", result.to_csv(index=False).encode("utf-8"),
                       file_name="hasil_prediksi_churn.csv", mime="text/csv")

    if "Churn_Probability" in result.columns:
        with st.expander("Verifikasi terhadap kolom Churn_Probability"):
            diff = (result["Skor_Risiko"] - result["Churn_Probability"]).abs().max()
            st.write(f"Selisih maksimum: {diff:.6f}")


DASHBOARD_URL = "https://public.tableau.com/views/CustomerChurn--E-Commerce/Overview"


def page_about():
    st.title("Tentang Model")

    st.subheader("Model dan metrik utama")
    st.markdown(
        "Model final adalah **XGBoost** hasil *hyperparameter tuning*, dipilih dari lima algoritma yang "
        "dibandingkan dengan *5-fold cross-validation*. Metrik utamanya adalah **F3**, yang memberi bobot "
        "*recall* sekitar 9x lebih besar daripada *precision*: kehilangan satu pelanggan (CLV sekitar \\$200) "
        "kira-kira 10x lebih mahal daripada satu penawaran retensi yang meleset (sekitar \\$20)."
    )

    st.subheader("Performa pada data test (1,015 pelanggan)")
    p1, p2, p3 = st.columns(3)
    p1.metric("Recall", "92.9%")
    p2.metric("Precision", "83.0%")
    p3.metric("F3", "0.9176")
    st.table(pd.DataFrame(
        {"Diprediksi tidak churn": [815, 12], "Diprediksi churn": [32, 156]},
        index=["Aktual tidak churn", "Aktual churn"],
    ))
    st.caption(
        "Dari 168 pelanggan yang benar-benar churn, 156 terdeteksi dan 12 terlewat; "
        "32 pelanggan ditandai berisiko padahal tidak churn."
    )

    st.subheader("Cara membaca skor dan Risk Tier")
    st.markdown(
        "Skor risiko (0-100%) adalah keluaran model, **bukan probabilitas yang terkalibrasi**: "
        "pembobotan kelas membuat skor cenderung lebih tinggi daripada tingkat churn sebenarnya. "
        "Karena itu, baca skor melalui *Risk Tier* berikut, yang diukur pada data test."
    )
    st.table(pd.DataFrame({
        "Risk Tier": ["Tinggi", "Sedang", "Rendah"],
        "Rentang skor": ["> 0.7", "0.3 - 0.7", "< 0.3"],
        "Pelanggan (data test)": [155, 90, 770],
        "Churn aktual": ["93.55%", "15.56%", "1.17%"],
        "Saran": ["Prioritaskan retensi", "Opsional, pantau", "Tidak perlu intervensi"],
    }).set_index("Risk Tier"))

    st.subheader("Faktor yang paling berpengaruh")
    st.markdown(
        "- **Tenure** dan **Complain** adalah dua faktor terpenting menurut dua metode interpretasi (*gain* dan SHAP). "
        "Pelanggan pada 0-1 bulan pertama churn sekitar 51%, dan pelanggan yang pernah komplain churn sekitar 31% "
        "(vs 11% yang tidak).\n"
        "- Menurut SHAP, **NumberOfAddress** dan **CashbackAmount** menyusul di posisi ketiga dan keempat."
    )

    st.subheader("Dampak bisnis (data test)")
    st.markdown(
        "Estimasi biaya kesalahan prediksi (churn terlewat x \\$200 + penawaran sia-sia x \\$20) adalah "
        "**\\$3,040** dengan model, dibandingkan **\\$16,940** jika semua pelanggan ditawari retensi, "
        "sehingga penghematan sekitar **\\$13,900 (82.1%)**. Angka ini mengasumsikan setiap penawaran "
        "berhasil mempertahankan pelanggan."
    )

    st.subheader("Batasan")
    st.markdown(
        "- Definisi churn pada data latih tidak dijelaskan oleh sumber data; model memperlakukannya apa adanya.\n"
        "- Paling andal untuk pelanggan dengan profil serupa data historis; perlu dipantau dan dilatih ulang berkala.\n"
        "- CLV dan biaya retensi memakai benchmark industri global dalam USD, bukan data perusahaan sebenarnya.\n"
        "- Evaluasi memakai 1,015 pelanggan data test, sehingga hasil pada data baru dapat berbeda.\n"
        "- Gunakan sebagai alat bantu prioritas, bukan keputusan otomatis."
    )
    st.markdown(f"[Lihat dashboard Tableau]({DASHBOARD_URL})")


PAGES = {
    "Beranda": page_home,
    "Prediksi Individu": page_individual,
    "Prediksi Batch": page_batch,
    "Tentang Model": page_about,
}

st.sidebar.title("Navigasi")
choice = st.sidebar.radio("Pilih halaman", list(PAGES))
st.sidebar.caption("Model: XGBoost (hasil tuning), dilatih pada data pelanggan e-commerce.")
PAGES[choice]()
