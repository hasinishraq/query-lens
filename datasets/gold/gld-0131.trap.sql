SELECT COUNT(*) FROM payments p JOIN orders o ON o.order_id = p.order_id WHERE p.method = 'card' AND NOT o.is_deleted AND CAST(o.order_at AS DATE) BETWEEN DATE '2024-12-01' AND DATE '2024-12-31'
