SELECT COUNT(*) AS n FROM orders o WHERE NOT o.is_deleted AND o.status IN ('cancelled', 'canceled') AND CAST(o.order_at AS DATE) BETWEEN DATE '2025-03-01' AND DATE '2025-03-31'
