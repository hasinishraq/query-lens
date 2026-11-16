SELECT p.product_id, (SELECT name FROM sellers x WHERE x.seller_id = p.seller_id), p.list_price_bdt FROM products p WHERE p.name = 'Classic Panjabi'
