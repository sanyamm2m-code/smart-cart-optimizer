"""
smart_cart.py
-------------
All-in-one Smart Cart Optimizer.
Combines: catalog loader, knapsack DP, ML recommender, terminal UI.

Run with:
    python smart_cart.py

Make sure catalog.json and order_history.csv are in the same folder.
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# =============================================================================
# SECTION 1 — CATALOG LOADER
# =============================================================================

def load_catalog():
    path = os.path.join(os.path.dirname(__file__), "catalog.json")
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# SECTION 2 — KNAPSACK OPTIMIZER
# =============================================================================

def knapsack(items, gap):
    """
    0/1 Knapsack DP.
    Finds the combination of items whose total price is <= gap
    and as close to gap as possible.
    """
    if gap <= 0:
        return [], 0

    n = len(items)
    dp = [[0] * (gap + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        item_price = items[i - 1]["price"]
        for w in range(gap + 1):
            dp[i][w] = dp[i - 1][w]
            if item_price <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - item_price] + item_price)

    # Traceback to find which items were selected
    selected_items = []
    w = gap
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(items[i - 1])
            w -= items[i - 1]["price"]

    total_price = sum(item["price"] for item in selected_items)
    return selected_items, total_price


def get_affordable_items(cash, cart_total, catalog, category_filter=None):
    """Returns all items priced <= gap, sorted cheapest first."""
    gap = int(cash - cart_total)
    if gap <= 0:
        return [], gap

    filtered = catalog
    if category_filter:
        filtered = [item for item in catalog if item["category"] in category_filter]

    affordable = [item for item in filtered if item["price"] <= gap]
    affordable.sort(key=lambda x: x["price"])
    return affordable, gap


def validate_selection(selected_ids, cash, cart_total, catalog):
    """Checks if user-picked items fit within the gap."""
    gap = int(cash - cart_total)
    id_map = {item["id"]: item for item in catalog}

    selected_items = [id_map[i] for i in selected_ids if i in id_map]
    selection_total = sum(item["price"] for item in selected_items)

    return {
        "valid": selection_total <= gap,
        "selected_items": selected_items,
        "selection_total": selection_total,
        "leftover": gap - selection_total,
        "new_cart_total": cart_total + selection_total
    }


# =============================================================================
# SECTION 3 — ML RECOMMENDER (KNN + Cosine Similarity)
# =============================================================================

def load_history():
    path = os.path.join(os.path.dirname(__file__), "order_history.csv")
    return pd.read_csv(path)


def build_user_category_matrix(df):
    """
    Rows = users, Columns = categories, Values = purchase frequency.
    Tells us how often each user buys from each category.
    """
    return df.groupby(["user_id", "category"]).size().unstack(fill_value=0)


class CartRecommender:
    def __init__(self, k=3):
        self.k = k
        self._load_and_fit()

    def _load_and_fit(self):
        df = load_history()
        self.matrix = build_user_category_matrix(df)
        self.categories = list(self.matrix.columns)

        # Normalize so users with more orders don't dominate
        self.normalized = normalize(self.matrix.values, norm="l2")

        # KNN with cosine similarity
        self.model = NearestNeighbors(
            n_neighbors=min(self.k + 1, len(self.matrix)),
            metric="cosine",
            algorithm="brute"
        )
        self.model.fit(self.normalized)

    def _get_category_scores(self, user_id):
        """
        For a known user: find similar users via KNN,
        average their category preferences, return as scores.
        For unknown user: return equal scores (no personalization).
        """
        if user_id not in self.matrix.index:
            return {cat: 1.0 for cat in self.categories}

        vec = self.matrix.loc[user_id].values.astype(float)
        norm = np.linalg.norm(vec)
        user_vec = vec / norm if norm > 0 else vec

        distances, indices = self.model.kneighbors([user_vec])

        # Skip the user themselves (distance ~0)
        neighbor_indices = [
            idx for dist, idx in zip(distances[0], indices[0])
            if self.matrix.index[idx] != user_id
        ][:self.k]

        if not neighbor_indices:
            return {cat: 1.0 for cat in self.categories}

        avg_profile = self.normalized[neighbor_indices].mean(axis=0)
        return {cat: float(avg_profile[i]) for i, cat in enumerate(self.categories)}

    def rerank(self, user_id, candidate_items):
        """Reranks items by how much this user's neighbors prefer that category."""
        if not candidate_items:
            return []

        scores = self._get_category_scores(user_id)
        scored = [(item, scores.get(item["category"], 0.5)) for item in candidate_items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored]


# =============================================================================
# SECTION 4 — TERMINAL UI
# =============================================================================

PROFILE_LABELS = {
    "u1": "Snack Lover",
    "u2": "Health Focused",
    "u3": "Home Cook",
    "u4": "Student",
    "u5": "Sweet Tooth",
    "u6": "Household Head",
}


def print_divider():
    print("-" * 55)


def pick_profile():
    print("\nSelect your profile:")
    for key, label in PROFILE_LABELS.items():
        print(f"  [{key}] {label}")
    while True:
        choice = input("Enter profile (u1-u6): ").strip().lower()
        if choice in PROFILE_LABELS:
            return choice
        print("  Invalid. Enter u1 to u6.")


def show_items(items):
    print_divider()
    print(f"  {'ID':<5} {'Item':<35} {'Category':<15} {'Price'}")
    print_divider()
    for item in items:
        print(f"  [{item['id']:<3}] {item['name']:<35} {item['category']:<15} ₹{item['price']}")
    print_divider()


def pick_items(affordable, gap):
    """User picks items by ID, running total updates after each pick."""
    selected_ids = []
    running_total = 0

    print(f"\nGap to fill: ₹{gap}")
    print("Enter item IDs one at a time. Enter 0 or leave blank to finish.\n")

    while True:
        remaining = gap - running_total
        try:
            raw = input(f"  Add item ID [₹{remaining} remaining]: ").strip()
        except EOFError:
            break

        if raw == "0" or raw == "":
            break

        if not raw.isdigit():
            print("  Enter a valid numeric item ID.")
            continue

        item_id = int(raw)
        match = next((i for i in affordable if i["id"] == item_id), None)

        if not match:
            print(f"  Item {item_id} not in the affordable list.")
            continue
        if item_id in selected_ids:
            print(f"  '{match['name']}' already added.")
            continue
        if match["price"] > remaining:
            print(f"  '{match['name']}' costs ₹{match['price']} but only ₹{remaining} remains.")
            continue

        selected_ids.append(item_id)
        running_total += match["price"]
        print(f"  ✓ Added '{match['name']}' — ₹{match['price']} | Total so far: ₹{running_total}")

    return selected_ids


def main():
    print("\n" + "=" * 55)
    print("        SMART CART OPTIMIZER")
    print("=" * 55)

    # Step 1 — get cash and cart total
    try:
        cash = int(input("\nEnter cash in hand (₹): ").strip())
        cart_total = int(input("Enter current cart total (₹): ").strip())
    except ValueError:
        print("Please enter valid numbers.")
        return

    # Step 2 — check gap
    gap = cash - cart_total
    if gap <= 0:
        print(f"\nCart total (₹{cart_total}) already meets or exceeds your cash (₹{cash}).")
        return
    print(f"\nGap to fill: ₹{gap}")

    # Step 3 — load catalog, get affordable items
    catalog = load_catalog()
    affordable, _ = get_affordable_items(cash, cart_total, catalog)

    if not affordable:
        print(f"No items available within ₹{gap}.")
        return

    # Step 4 — pick profile, ML rerank
    user_id = pick_profile()
    print(f"\nLoading recommendations for '{PROFILE_LABELS[user_id]}'...")

    recommender = CartRecommender(k=3)
    reranked = recommender.rerank(user_id, affordable)

    print("\nItems you can add (sorted by your preferences):")
    show_items(reranked)

    # Step 5 — user picks items
    selected_ids = pick_items(reranked, gap)

    if not selected_ids:
        print("\nNo items selected. Exiting.")
        return

    # Step 6 — validate and show summary
    result = validate_selection(selected_ids, cash, cart_total, catalog)

    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    print(f"  Cash in hand    : ₹{cash}")
    print(f"  Original cart   : ₹{cart_total}")
    print(f"  Items added     :")
    for item in result["selected_items"]:
        print(f"    + {item['name']} — ₹{item['price']}")
    print(f"  Items total     : ₹{result['selection_total']}")
    print(f"  New cart total  : ₹{result['new_cart_total']}")
    print(f"  Leftover cash   : ₹{result['leftover']}")

    if result["valid"]:
        print("\n  ✅ Within budget!")
    else:
        print(f"\n  ❌ Over budget by ₹{abs(result['leftover'])}.")

    print("=" * 55)


if __name__ == "__main__":
    main()
