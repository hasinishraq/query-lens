SELECT p.product_id, s.name, p.list_price_bdt FROM products p JOIN sellers s ON s.seller_id = p.seller_id WHERE p.name = 'Classic Panjabi'
