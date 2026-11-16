SELECT COUNT(product_id) AS n FROM products WHERE is_deleted = FALSE AND category_id IN ( SELECT category_id FROM categories WHERE level = 3 AND path LIKE 'Sports & Outdoors > %')
