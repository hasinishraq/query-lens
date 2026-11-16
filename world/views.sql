-- Role databases (CLAUDE.md section 6.7), MySQL 8 dialect. Applied by the generator after the base tables are
-- loaded. Placeholders: {base} = bazaar_<size>, {analyst}/{support}/{finance} = bazaar_<size>_<role>.
-- Each role user has SELECT + SHOW VIEW on its own role database only, so base tables and columns that are not
-- in these views cannot be reached. The SQL guard enforces the same rules again in code.

-- =============================================================== analyst
CREATE VIEW {analyst}.regions AS SELECT * FROM {base}.regions;
CREATE VIEW {analyst}.cities AS SELECT * FROM {base}.cities;
CREATE VIEW {analyst}.calendar_dim AS SELECT * FROM {base}.calendar_dim;
CREATE VIEW {analyst}.currency_rates AS SELECT * FROM {base}.currency_rates;
-- customers: no name, email, phone, gender, birth year; signup reduced to its month
CREATE VIEW {analyst}.customers AS
  SELECT customer_id, city_id, CAST(DATE_FORMAT(signup_at, '%Y-%m-01') AS DATE) AS signup_month, signup_channel,
         acquisition_source, is_verified, preferred_language, is_deleted
  FROM {base}.customers;
CREATE VIEW {analyst}.customer_addresses AS
  SELECT address_id, customer_id, city_id, address_type, is_default, created_at, is_deleted
  FROM {base}.customer_addresses;
CREATE VIEW {analyst}.customer_segment_history AS SELECT * FROM {base}.customer_segment_history;
CREATE VIEW {analyst}.loyalty_accounts AS SELECT * FROM {base}.loyalty_accounts;
CREATE VIEW {analyst}.loyalty_transactions AS SELECT * FROM {base}.loyalty_transactions;
CREATE VIEW {analyst}.categories AS SELECT * FROM {base}.categories;
CREATE VIEW {analyst}.brands AS SELECT * FROM {base}.brands;
CREATE VIEW {analyst}.products AS SELECT * FROM {base}.products;
CREATE VIEW {analyst}.product_variants AS SELECT * FROM {base}.product_variants;
CREATE VIEW {analyst}.product_price_history AS SELECT * FROM {base}.product_price_history;
CREATE VIEW {analyst}.sellers AS
  SELECT seller_id, name, legal_name, city_id, joined_at, seller_type, status, rating_avg FROM {base}.sellers;
CREATE VIEW {analyst}.seller_tier_history AS SELECT * FROM {base}.seller_tier_history;
CREATE VIEW {analyst}.seller_payouts AS SELECT * FROM {base}.seller_payouts;
CREATE VIEW {analyst}.warehouses AS SELECT * FROM {base}.warehouses;
CREATE VIEW {analyst}.inventory_snapshots AS SELECT * FROM {base}.inventory_snapshots;
CREATE VIEW {analyst}.orders AS SELECT * FROM {base}.orders;
CREATE VIEW {analyst}.order_items AS SELECT * FROM {base}.order_items;
CREATE VIEW {analyst}.order_status_events AS SELECT * FROM {base}.order_status_events;
CREATE VIEW {analyst}.payments AS SELECT * FROM {base}.payments;
CREATE VIEW {analyst}.refunds AS SELECT * FROM {base}.refunds;
CREATE VIEW {analyst}.returns AS SELECT * FROM {base}.returns;
CREATE VIEW {analyst}.couriers AS SELECT * FROM {base}.couriers;
CREATE VIEW {analyst}.shipments AS SELECT * FROM {base}.shipments;
CREATE VIEW {analyst}.shipment_events AS SELECT * FROM {base}.shipment_events;
CREATE VIEW {analyst}.marketing_campaigns AS SELECT * FROM {base}.marketing_campaigns;
CREATE VIEW {analyst}.promotions AS SELECT * FROM {base}.promotions;
CREATE VIEW {analyst}.promotion_redemptions AS SELECT * FROM {base}.promotion_redemptions;
CREATE VIEW {analyst}.web_sessions AS SELECT * FROM {base}.web_sessions;
CREATE VIEW {analyst}.support_tickets AS SELECT * FROM {base}.support_tickets;
CREATE VIEW {analyst}.product_reviews AS SELECT * FROM {base}.product_reviews;
-- (no analyst support_messages)

-- =============================================================== support
CREATE VIEW {support}.regions AS SELECT * FROM {base}.regions;
CREATE VIEW {support}.cities AS SELECT * FROM {base}.cities;
CREATE VIEW {support}.calendar_dim AS SELECT * FROM {base}.calendar_dim;
CREATE VIEW {support}.customers AS
  SELECT customer_id, full_name, city_id, signup_at, preferred_language, is_verified, is_deleted
  FROM {base}.customers;
CREATE VIEW {support}.customer_addresses AS
  SELECT address_id, customer_id, city_id, address_type, is_default, is_deleted FROM {base}.customer_addresses;
CREATE VIEW {support}.products AS
  SELECT product_id, seller_id, name, category_id, status, is_deleted FROM {base}.products;
CREATE VIEW {support}.product_variants AS SELECT variant_id, product_id, sku, size, colour FROM {base}.product_variants;
CREATE VIEW {support}.sellers AS SELECT seller_id, name, city_id, status FROM {base}.sellers;
CREATE VIEW {support}.orders AS SELECT * FROM {base}.orders;
CREATE VIEW {support}.order_items AS SELECT * FROM {base}.order_items;
CREATE VIEW {support}.order_status_events AS SELECT * FROM {base}.order_status_events;
CREATE VIEW {support}.payments AS SELECT * FROM {base}.payments;
CREATE VIEW {support}.refunds AS SELECT * FROM {base}.refunds;
CREATE VIEW {support}.returns AS SELECT * FROM {base}.returns;
CREATE VIEW {support}.couriers AS SELECT * FROM {base}.couriers;
CREATE VIEW {support}.shipments AS SELECT * FROM {base}.shipments;
CREATE VIEW {support}.shipment_events AS SELECT * FROM {base}.shipment_events;
CREATE VIEW {support}.support_tickets AS SELECT * FROM {base}.support_tickets;
CREATE VIEW {support}.support_messages AS SELECT * FROM {base}.support_messages;

-- =============================================================== finance
CREATE VIEW {finance}.calendar_dim AS SELECT * FROM {base}.calendar_dim;
CREATE VIEW {finance}.currency_rates AS SELECT * FROM {base}.currency_rates;
CREATE VIEW {finance}.orders AS
  SELECT order_id, order_at, status, channel, currency, total_amount, total_bdt, promo_id, is_deleted, updated_at
  FROM {base}.orders;
CREATE VIEW {finance}.payments AS SELECT * FROM {base}.payments;
CREATE VIEW {finance}.refunds AS SELECT * FROM {base}.refunds;
CREATE VIEW {finance}.returns AS SELECT * FROM {base}.returns;
CREATE VIEW {finance}.sellers AS SELECT seller_id, name, seller_type, status FROM {base}.sellers;
CREATE VIEW {finance}.seller_payouts AS SELECT * FROM {base}.seller_payouts;
