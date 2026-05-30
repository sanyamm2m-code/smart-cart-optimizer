"""
app.py
------
Day 3 — Streamlit UI for Smart Cart Optimizer.
Run with: python -m streamlit run app.py

Make sure catalog.json, order_history.csv and smart_cart.py are in the same folder.
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smart_cart import (
    load_catalog,
    get_affordable_items,
    validate_selection,
    CartRecommender,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Smart Cart Optimizer",
    page_icon="🛒",
    layout="wide",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.main { background-color: #0f0f0f; }

.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #f0f0f0;
}
.metric-value.accent { color: #c8f135; }
.metric-value.danger { color: #ff4f4f; }

.gap-bar-bg {
    background: #2a2a2a;
    border-radius: 8px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin-top: 8px;
}
.gap-bar-fill {
    height: 12px;
    border-radius: 8px;
    transition: width 0.4s ease;
}

.summary-box {
    background: #141414;
    border: 1px solid #c8f135;
    border-radius: 14px;
    padding: 24px 28px;
}

.app-header {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: #f0f0f0;
    line-height: 1;
    margin-bottom: 4px;
}
.app-sub {
    color: #666;
    font-size: 14px;
    margin-bottom: 32px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_resource
def get_recommender():
    return CartRecommender(k=3)

@st.cache_data
def get_catalog():
    return load_catalog()

catalog = get_catalog()
recommender = get_recommender()

PROFILE_LABELS = {
    "u1": "🍿 Snack Lover",
    "u2": "🥗 Health Focused",
    "u3": "🍳 Home Cook",
    "u4": "📚 Student",
    "u5": "🍫 Sweet Tooth",
    "u6": "🏠 Household Head",
}

ALL_CATEGORIES = sorted(set(item["category"] for item in catalog))

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<div class="app-header">🛒 Smart Cart Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Fill your budget gap with personalized item suggestions</div>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Your Details")
    cash = st.number_input("Cash in hand (₹)", min_value=1, value=500, step=10)
    cart_total = st.number_input("Current cart total (₹)", min_value=0, value=443, step=10)
    st.markdown("---")
    profile_key = st.selectbox("Your profile", list(PROFILE_LABELS.keys()),
                                format_func=lambda x: PROFILE_LABELS[x])
    st.markdown("---")
    st.markdown("### 🗂️ Filter by Category")
    selected_categories = st.multiselect(
        "Show only these categories",
        ALL_CATEGORIES,
        default=ALL_CATEGORIES,
    )

# =============================================================================
# GAP + METRICS
# =============================================================================

gap = int(cash - cart_total)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Cash in Hand</div>
        <div class="metric-value">₹{cash}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Cart Total</div>
        <div class="metric-value">₹{cart_total}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    gap_class = "accent" if gap > 0 else "danger"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Gap to Fill</div>
        <div class="metric-value {gap_class}">₹{gap}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Profile</div>
        <div class="metric-value" style="font-size:22px">{PROFILE_LABELS[profile_key]}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if gap <= 0:
    st.warning("⚠️ Your cart total is already at or above your cash.")
    st.stop()

# =============================================================================
# AFFORDABLE ITEMS + RERANK
# =============================================================================

affordable, _ = get_affordable_items(cash, cart_total, catalog,
                                      category_filter=selected_categories if selected_categories else None)

if not affordable:
    st.error(f"No items found within ₹{gap} in the selected categories.")
    st.stop()

reranked = recommender.rerank(profile_key, affordable)

# =============================================================================
# MAIN LAYOUT
# =============================================================================

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### 🧾 Available Items")
    st.caption(f"{len(reranked)} items within ₹{gap} — sorted by your preference")

    item_options = {
        f"{item['name']}  ·  ₹{item['price']}  [{item['category']}]": item
        for item in reranked
    }

    chosen_labels = st.multiselect(
        "Select items to add to your cart",
        list(item_options.keys()),
        help="Items are sorted by your profile preference. Pick any combination within the gap."
    )

    chosen_items = [item_options[label] for label in chosen_labels]
    running_total = sum(item["price"] for item in chosen_items)
    remaining = gap - running_total
    fill_pct = min(running_total / gap * 100, 100) if gap > 0 else 0
    bar_color = "#ff4f4f" if running_total > gap else "#c8f135"

    st.markdown(f"""
    <div style="margin-top:16px">
        <div style="display:flex; justify-content:space-between; font-size:13px; color:#888; margin-bottom:4px">
            <span>Budget used: <b style="color:#f0f0f0">₹{running_total}</b></span>
            <span>Remaining: <b style="color:{bar_color}">₹{remaining}</b></span>
        </div>
        <div class="gap-bar-bg">
            <div class="gap-bar-fill" style="width:{fill_pct}%; background:{bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if running_total > gap:
        st.error(f"❌ Over budget by ₹{running_total - gap}. Please remove an item.")

with right:
    st.markdown("### 📊 Summary")

    if chosen_items:
        result = validate_selection([i["id"] for i in chosen_items], cash, cart_total, catalog)

        st.markdown(f"""
        <div class="summary-box">
            <div style="font-family:'Syne',sans-serif; font-size:13px; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px">Cart Breakdown</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:14px; color:#aaa">
                <span>Original cart</span><span style="color:#f0f0f0">₹{cart_total}</span>
            </div>
        """, unsafe_allow_html=True)

        for item in chosen_items:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:13px; color:#888; padding-left:8px; border-left:2px solid #c8f135;">
                <span>+ {item['name']}</span><span>₹{item['price']}</span>
            </div>
            """, unsafe_allow_html=True)

        status_color = "#c8f135" if result["valid"] else "#ff4f4f"
        status_icon = "✅" if result["valid"] else "❌"

        st.markdown(f"""
            <div style="border-top:1px solid #2a2a2a; margin:12px 0; padding-top:12px; display:flex; justify-content:space-between; font-family:'Syne',sans-serif; font-size:18px; font-weight:700; color:#f0f0f0">
                <span>New Total</span><span style="color:{status_color}">₹{result['new_cart_total']}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:13px; color:#888">
                <span>Leftover cash</span><span>₹{result['leftover']}</span>
            </div>
            <div style="margin-top:14px; font-size:15px">{status_icon} {'Within budget!' if result['valid'] else f"Over by ₹{abs(result['leftover'])}"}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cat_counts = {}
        for item in chosen_items:
            cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + item["price"]

        if cat_counts:
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#141414")
            ax.set_facecolor("#141414")
            colors = ["#c8f135", "#7ecb4b", "#f1a735", "#f15535", "#35a9f1", "#a835f1", "#f135a9"]
            wedges, texts, autotexts = ax.pie(
                cat_counts.values(),
                labels=cat_counts.keys(),
                autopct="%1.0f%%",
                colors=colors[:len(cat_counts)],
                textprops={"color": "#aaa", "fontsize": 10},
                wedgeprops={"linewidth": 2, "edgecolor": "#141414"}
            )
            for at in autotexts:
                at.set_color("#141414")
                at.set_fontweight("bold")
            ax.set_title("Spend by Category", color="#888", fontsize=11, pad=10)
            st.pyplot(fig)
            plt.close()
    else:
        st.markdown("""
        <div style="color:#555; font-size:14px; text-align:center; padding:40px 0;">
            ← Select items from the left<br>to see your summary here
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# CONFIRM BUTTON
# =============================================================================

st.markdown("---")

if chosen_items:
    result = validate_selection([i["id"] for i in chosen_items], cash, cart_total, catalog)
    if result["valid"]:
        if st.button("✅ Confirm & Add to Cart", type="primary", use_container_width=True):
            st.balloons()
            st.success(f"🎉 Added {len(chosen_items)} item(s)! New cart total: ₹{result['new_cart_total']}  |  Leftover: ₹{result['leftover']}")
    else:
        st.button("❌ Over Budget — Remove an Item", disabled=True, use_container_width=True)
else:
    st.button("Select items above to continue", disabled=True, use_container_width=True)