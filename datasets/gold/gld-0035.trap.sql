SELECT AVG(CASE WHEN o.status = 'cancelled' THEN 1.0 ELSE 0.0 END) AS value FROM orders o WHERE NOT o.is_deleted AND CAST(o.order_at AS DATE) BETWEEN DATE '2025-06-01' AND DATE '2025-06-30'
