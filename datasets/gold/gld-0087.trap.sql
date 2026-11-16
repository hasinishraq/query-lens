SELECT COUNT(*) AS n FROM orders o WHERE NOT o.is_deleted AND o.status = 'cancelled' AND CAST(o.order_at AS DATE) BETWEEN DATE '2024-08-01' AND DATE '2024-08-31'
