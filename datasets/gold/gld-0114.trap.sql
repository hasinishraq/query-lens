SELECT ph.price_bdt FROM product_price_history ph JOIN product_variants v ON v.variant_id = ph.variant_id WHERE v.sku = 'BZ-00033-000077' AND ph.valid_to = DATE '9999-12-31'
