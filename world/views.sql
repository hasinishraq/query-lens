-- Role schemas (CLAUDE.md section 6.7). Applied by the generator after the base tables are loaded.
-- The agent connects read-only with search_path set to its role schema; the guard additionally
-- rejects references to any other schema and to columns outside the role view.

-- =============================================================== v_analyst
CREATE SCHEMA v_analyst;
CREATE VIEW v_analyst.regions AS SELECT * FROM main.regions;
CREATE VIEW v_analyst.cities AS SELECT * FROM main.cities;
CREATE VIEW v_analyst.calendar_dim AS SELECT * FROM main.calendar_dim;
CREATE VIEW v_analyst.currency_rates AS SELECT * FROM main.currency_rates;
-- customers: no name, email, phone, gender, birth year; signup reduced to its month
CREATE VIEW v_analyst.customers AS
  SELECT customer_id, city_id, CAST(date_trunc('month', signup_at) AS DATE) AS signup_month, signup_channel,
         acquisition_source, is_verified, preferred_language, is_deleted
  FROM main.customers;
CREATE VIEW v_analyst.customer_addresses AS
  SELECT address_id, customer_id, city_id, address_type, is_default, created_at, is_deleted
  FROM main.customer_addresses;
CREATE VIEW v_analyst.customer_segment_history AS SELECT * FROM main.customer_segment_history;
CREATE VIEW v_analyst.loyalty_accounts AS SELECT * FROM main.loyalty_accounts;
CREATE VIEW v_analyst.loyalty_transactions AS SELECT * FROM main.loyalty_transactions;
CREATE VIEW v_analyst.categories AS SELECT * FROM main.categories;
CREATE VIEW v_analyst.brands AS SELECT * FROM main.brands;
CREATE VIEW v_analyst.products AS SELECT * FROM main.products;
CREATE VIEW v_analyst.product_variants AS SELECT * FROM main.product_variants;
CREATE VIEW v_analyst.product_price_history AS SELECT * FROM main.product_price_history;
CREATE VIEW v_analyst.sellers AS
  SELECT seller_id, name, legal_name, city_id, joined_at, seller_type, status, rating_avg FROM main.sellers;
CREATE VIEW v_analyst.seller_tier_history AS SELECT * FROM main.seller_tier_history;
CREATE VIEW v_analyst.seller_payouts AS SELECT * FROM main.seller_payouts;
CREATE VIEW v_analyst.warehouses AS SELECT * FROM main.warehouses;
CREATE VIEW v_analyst.inventory_snapshots AS SELECT * FROM main.inventory_snapshots;
CREATE VIEW v_analyst.orders AS SELECT * FROM main.orders;
CREATE VIEW v_analyst.order_items AS SELECT * FROM main.order_items;
CREATE VIEW v_analyst.order_status_events AS SELECT * FROM main.order_status_events;
CREATE VIEW v_analyst.payments AS SELECT * FROM main.payments;
CREATE VIEW v_analyst.refunds AS SELECT * FROM main.refunds;
CREATE VIEW v_analyst.returns AS SELECT * FROM main.returns;
CREATE VIEW v_analyst.couriers AS SELECT * FROM main.couriers;
CREATE VIEW v_analyst.shipments AS SELECT * FROM main.shipments;
CREATE VIEW v_analyst.shipment_events AS SELECT * FROM main.shipment_events;
CREATE VIEW v_analyst.marketing_campaigns AS SELECT * FROM main.marketing_campaigns;
CREATE VIEW v_analyst.promotions AS SELECT * FROM main.promotions;
CREATE VIEW v_analyst.promotion_redemptions AS SELECT * FROM main.promotion_redemptions;
CREATE VIEW v_analyst.web_sessions AS SELECT * FROM main.web_sessions;
CREATE VIEW v_analyst.support_tickets AS SELECT * FROM main.support_tickets;
CREATE VIEW v_analyst.product_reviews AS SELECT * FROM main.product_reviews;
-- (no v_analyst.support_messages)

-- =============================================================== v_support
CREATE SCHEMA v_support;
CREATE VIEW v_support.regions AS SELECT * FROM main.regions;
CREATE VIEW v_support.cities AS SELECT * FROM main.cities;
CREATE VIEW v_support.calendar_dim AS SELECT * FROM main.calendar_dim;
CREATE VIEW v_support.customers AS
  SELECT customer_id, full_name, city_id, signup_at, preferred_language, is_verified, is_deleted
  FROM main.customers;
CREATE VIEW v_support.customer_addresses AS
  SELECT address_id, customer_id, city_id, address_type, is_default, is_deleted FROM main.customer_addresses;
CREATE VIEW v_support.products AS
  SELECT product_id, seller_id, name, category_id, status, is_deleted FROM main.products;
CREATE VIEW v_support.product_variants AS SELECT variant_id, product_id, sku, size, colour FROM main.product_variants;
CREATE VIEW v_support.sellers AS SELECT seller_id, name, city_id, status FROM main.sellers;
CREATE VIEW v_support.orders AS SELECT * FROM main.orders;
CREATE VIEW v_support.order_items AS SELECT * FROM main.order_items;
CREATE VIEW v_support.order_status_events AS SELECT * FROM main.order_status_events;
CREATE VIEW v_support.payments AS SELECT * FROM main.payments;
CREATE VIEW v_support.refunds AS SELECT * FROM main.refunds;
CREATE VIEW v_support.returns AS SELECT * FROM main.returns;
CREATE VIEW v_support.couriers AS SELECT * FROM main.couriers;
CREATE VIEW v_support.shipments AS SELECT * FROM main.shipments;
CREATE VIEW v_support.shipment_events AS SELECT * FROM main.shipment_events;
CREATE VIEW v_support.support_tickets AS SELECT * FROM main.support_tickets;
CREATE VIEW v_support.support_messages AS SELECT * FROM main.support_messages;

-- =============================================================== v_finance
CREATE SCHEMA v_finance;
CREATE VIEW v_finance.calendar_dim AS SELECT * FROM main.calendar_dim;
CREATE VIEW v_finance.currency_rates AS SELECT * FROM main.currency_rates;
CREATE VIEW v_finance.orders AS
  SELECT order_id, order_at, status, channel, currency, total_amount, total_bdt, promo_id, is_deleted, updated_at
  FROM main.orders;
CREATE VIEW v_finance.payments AS SELECT * FROM main.payments;
CREATE VIEW v_finance.refunds AS SELECT * FROM main.refunds;
CREATE VIEW v_finance.returns AS SELECT * FROM main.returns;
CREATE VIEW v_finance.sellers AS SELECT seller_id, name, seller_type, status FROM main.sellers;
CREATE VIEW v_finance.seller_payouts AS SELECT * FROM main.seller_payouts;
