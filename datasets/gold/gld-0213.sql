SELECT signup_month, COUNT(*) FROM customers WHERE NOT is_deleted AND signup_month BETWEEN DATE '2026-01-01' AND DATE '2026-03-31' GROUP BY 1 ORDER BY 1
