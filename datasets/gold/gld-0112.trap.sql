SELECT ph.price_bdt FROM product_price_history ph JOIN product_variants v ON v.variant_id = ph.variant_id WHERE v.sku = 'BZ-00016-000039' AND ph.valid_to = DATE '9999-12-31'
