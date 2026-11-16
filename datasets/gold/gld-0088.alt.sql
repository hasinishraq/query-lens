SELECT COUNT(order_id) AS n FROM orders o WHERE o.is_deleted = FALSE AND o.status LIKE 'cancel%' AND o.order_at >= TIMESTAMP '2024-12-01 00:00:00' AND o.order_at < TIMESTAMP '2025-01-01 00:00:00'
