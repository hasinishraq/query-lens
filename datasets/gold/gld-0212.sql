SELECT signup_month, COUNT(*) FROM customers WHERE NOT is_deleted AND signup_month BETWEEN DATE '2025-07-01' AND DATE '2026-06-30' GROUP BY 1 ORDER BY 1
