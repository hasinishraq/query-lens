SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM products WHERE name = 'Classic Panjabi')
