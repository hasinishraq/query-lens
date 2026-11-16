SELECT MIN(p.product_id), MIN(s.name), AVG(p.list_price_bdt) FROM products p JOIN sellers s ON s.seller_id = p.seller_id WHERE p.name = 'Classic Panjabi'
