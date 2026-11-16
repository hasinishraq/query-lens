SELECT COUNT(order_id) AS n FROM orders o WHERE o.is_deleted = FALSE AND o.status LIKE 'cancel%' AND o.order_at >= TIMESTAMP '2024-08-01 00:00:00' AND o.order_at < TIMESTAMP '2024-09-01 00:00:00'
