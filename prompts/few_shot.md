<!-- prompt: few_shot | version: 1.1 -->
# Examples of good steps

Question: "Which region had the highest total revenue in 2025?"
run_python(purpose="Total 2025 revenue per region", code=
```
s = sales[sales["order_date"].dt.year == 2025]
revenue_by_region = s.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
print(revenue_by_region.iloc[0])
revenue_by_region
```)
final_answer(answer="North", key_findings=["North had the highest 2025 revenue: 1,234,567.89", ...])

Question: "Show the monthly trend of orders"
run_python(purpose="Monthly order counts and line chart", code=
```
monthly_orders = (sales.set_index("order_date").resample("MS").size()
                  .rename("orders").reset_index())
fig = px.line(monthly_orders, x="order_date", y="orders", markers=True,
              title="Orders per month", labels={"order_date": "Month", "orders": "Orders"})
show_chart(fig, "Orders per month")
monthly_orders.tail(12)
```)

Question: "Is there a significant difference in tip amount between card and cash payments?"
run_python(purpose="Welch t-test of tips by payment type", code=
```
card = trips.loc[trips["payment_type"] == "card", "tip_amount"].dropna()
cash = trips.loc[trips["payment_type"] == "cash", "tip_amount"].dropna()
t, p = stats.ttest_ind(card, cash, equal_var=False)
print(f"card mean={card.mean():.4f} n={len(card)}; cash mean={cash.mean():.4f} n={len(cash)}; t={t:.3f}; p={p:.3g}")
```)

Question (database): "Top 5 customers by number of orders"
run_sql(purpose="Count orders per customer", target="shop", result_name="top_customers", query=
```
SELECT c.name, COUNT(*) AS n_orders
FROM orders o JOIN customers c ON c.customer_id = o.customer_id
GROUP BY c.name ORDER BY n_orders DESC LIMIT 5
```)
