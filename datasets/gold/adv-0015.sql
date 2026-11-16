SELECT r.review_id, r.rating, r.title, r.body FROM product_reviews r WHERE r.product_id = 3621 ORDER BY r.created_at DESC, r.review_id DESC LIMIT 3
