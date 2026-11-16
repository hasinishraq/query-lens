SELECT COUNT(*) FROM (SELECT product_id FROM products WHERE name = 'Classic Panjabi' GROUP BY product_id) t
