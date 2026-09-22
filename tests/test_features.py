"""Comprehensive unit tests for catalog feature engineering including adversarial edge cases."""

import unittest
import numpy as np
import pandas as pd

from src.features import (
    CatalogFeatureExtractor,
    FoldSafeTargetEncoder,
    _extract_chunk,
    compute_text_surface_stats,
    extract_model_identifiers,
    extract_row_physical_specs,
    parse_battery_mah,
    parse_dimensions_cm,
    parse_pack_count,
    parse_percentage,
    parse_power_watts,
    parse_storage_gb,
    parse_voltage_volts,
    parse_volume_ml,
    parse_weight_grams,
)


class TestCatalogFeatureEngineering(unittest.TestCase):
    # -----------------------------------------------------------------------
    # Adversarial Edge Cases & Semantic Disambiguation Tests
    # -----------------------------------------------------------------------

    def test_cellular_5g_not_mistaken_for_weight(self):
        """Disambiguate telecommunication '5G'/'4G' from grams."""
        val_5g = parse_weight_grams("Samsung Galaxy Note 20 Ultra 5G Smartphone")
        self.assertIsNone(val_5g)

        val_4g = parse_weight_grams("4G LTE Wireless Mobile Router")
        self.assertIsNone(val_4g)

        # But legitimate small gram products must still work:
        val_saffron = parse_weight_grams("Pure Kashmiri Saffron 5 g")
        self.assertAlmostEqual(val_saffron, 5.0, places=1)

    def test_2d_dimensions_not_mistaken_for_multipack(self):
        """Disambiguate '10 x 20 cm' linear dimensions from '6 x 250 ml' multipacks."""
        cnt_frame, multi_frame = parse_pack_count("Wooden Photo Frame 10 x 20 cm")
        self.assertEqual(cnt_frame, 1.0)
        self.assertEqual(multi_frame, 0)

        cnt_mat, multi_mat = parse_pack_count("Yoga Mat 60 x 180 cm")
        self.assertEqual(cnt_mat, 1.0)
        self.assertEqual(multi_mat, 0)

        # Legitimate multipack must still be detected:
        cnt_drink, multi_drink = parse_pack_count("Energy Drink 6 x 250 ml")
        self.assertEqual(cnt_drink, 6.0)
        self.assertEqual(multi_drink, 1)

    def test_preposition_in_not_mistaken_for_inches(self):
        """Disambiguate '2 in 1' or 'in pouch' from '12 in' dimension."""
        dim_shampoo, _, _, _ = parse_dimensions_cm("Head & Shoulders 2 in 1 Shampoo 200 ml")
        self.assertIsNone(dim_shampoo)

        dim_combo, _, _, _ = parse_dimensions_cm("All in 1 Multi Surface Cleaner")
        self.assertIsNone(dim_combo)

        # Legitimate inches must still work:
        dim_pan, _, _, _ = parse_dimensions_cm("Stainless Steel Pan 12 in Heavy Duty")
        self.assertAlmostEqual(dim_pan, 30.48, places=2)

    def test_hierarchical_field_priority(self):
        """Title net weight must override ingredient/serving sizes in description."""
        title = "Gold Standard Whey Protein Tub 5 kg"
        desc = "Nutrition facts: 2g sugar, 500mg sodium, 24g protein per 30g scoop"
        specs = extract_row_physical_specs(title=title, pack=None, desc=desc)

        self.assertAlmostEqual(specs["unit_weight_g"], 5000.0, places=1)
        self.assertAlmostEqual(specs["total_weight_g"], 5000.0, places=1)

    # -----------------------------------------------------------------------
    # Pack Multiplier & Multipack Tests
    # -----------------------------------------------------------------------

    def test_multipack_dimension_format(self):
        """Test '6 x 250 ml' format required by study guide."""
        cnt, multi = parse_pack_count("Special Energy Drink 6 x 250 ml can")
        self.assertEqual(cnt, 6.0)
        self.assertEqual(multi, 1)

    def test_multipack_alternative_delimiters(self):
        cnt, multi = parse_pack_count("Organic Oats 2 * 500g pouch")
        self.assertEqual(cnt, 2.0)
        self.assertEqual(multi, 1)

    def test_pack_of_format(self):
        cnt, multi = parse_pack_count("Xplus Bath Loofah (Pack of 3)")
        self.assertEqual(cnt, 3.0)
        self.assertEqual(multi, 1)

    def test_set_and_combo_format(self):
        cnt, multi = parse_pack_count("Stainless Steel Cookware Set of 4")
        self.assertEqual(cnt, 4.0)
        self.assertEqual(multi, 1)

    def test_count_pack_shorthand(self):
        cnt, multi = parse_pack_count("Premium Cotton Socks 6-pack")
        self.assertEqual(cnt, 6.0)
        self.assertEqual(multi, 1)

        cnt_pcs, multi_pcs = parse_pack_count("Ballpoint Pens 10 pcs")
        self.assertEqual(cnt_pcs, 10.0)
        self.assertEqual(multi_pcs, 1)

    def test_compound_pack_multipliers(self):
        count, multi = parse_pack_count("Pack of 2 (3 x 100ml)")
        self.assertEqual((count, multi), (6.0, 1))

        count, multi = parse_pack_count("2 boxes of 6 pens")
        self.assertEqual((count, multi), (12.0, 1))

        specs = extract_row_physical_specs(
            "Energy Drink Pack of 2 (3 x 100ml)", "Pack of 2", None
        )
        self.assertEqual(specs["pack_count"], 6.0)
        self.assertEqual(specs["total_volume_ml"], 600.0)

    def test_pair_of_format(self):
        cnt, multi = parse_pack_count("Running Shoes (Pair of 2)")
        self.assertEqual(cnt, 2.0)
        self.assertEqual(multi, 1)

    def test_single_item_default(self):
        cnt, multi = parse_pack_count("Apple iPhone 13 Pro Max")
        self.assertEqual(cnt, 1.0)
        self.assertEqual(multi, 0)

    # -----------------------------------------------------------------------
    # Weight Extraction & Normalization Tests
    # -----------------------------------------------------------------------

    def test_weight_kg_format(self):
        """Test '1.5 kg' format required by study guide."""
        val = parse_weight_grams("Basmati Rice 1.5 kg")
        self.assertAlmostEqual(val, 1500.0, places=1)

    def test_weight_grams_and_gms(self):
        val1 = parse_weight_grams("Dark Chocolate Bar 500g")
        self.assertAlmostEqual(val1, 500.0, places=1)

        val2 = parse_weight_grams("Herbal Face Cream 250 gm")
        self.assertAlmostEqual(val2, 250.0, places=1)

    def test_weight_ounces_and_pounds(self):
        val_oz = parse_weight_grams("Ground Coffee 16 oz")
        self.assertAlmostEqual(val_oz, 453.59, places=1)

        val_lb = parse_weight_grams("Protein Powder 2.2 lbs")
        self.assertAlmostEqual(val_lb, 997.90, places=1)

    def test_weight_milligrams(self):
        val_mg = parse_weight_grams("Vitamin C Tablets 500 mg")
        self.assertAlmostEqual(val_mg, 0.5, places=3)

    def test_weight_fraction(self):
        val = parse_weight_grams("Organic Almond Flour 1/2 kg")
        self.assertAlmostEqual(val, 500.0, places=1)

    # -----------------------------------------------------------------------
    # Volume Extraction & Normalization Tests
    # -----------------------------------------------------------------------

    def test_volume_milliliters(self):
        val = parse_volume_ml("Aloe Vera Shampoo 250 ml")
        self.assertAlmostEqual(val, 250.0, places=1)

    def test_volume_liters(self):
        val = parse_volume_ml("Cold Pressed Olive Oil 1.5 L")
        self.assertAlmostEqual(val, 1500.0, places=1)

        val_ltr = parse_volume_ml("Mineral Water 2 ltr")
        self.assertAlmostEqual(val_ltr, 2000.0, places=1)

    def test_volume_fluid_ounces_and_gallons(self):
        val_floz = parse_volume_ml("Perfume Eau de Parfum 3.4 fl oz")
        self.assertAlmostEqual(val_floz, 100.55, places=1)

        val_gal = parse_volume_ml("Hand Sanitizer Refill 1 gallon")
        self.assertAlmostEqual(val_gal, 3785.41, places=1)

    def test_volume_fraction(self):
        val = parse_volume_ml("Fresh Milk 1/2 liter")
        self.assertAlmostEqual(val, 500.0, places=1)

    # -----------------------------------------------------------------------
    # Dimension Extraction & Normalization Tests
    # -----------------------------------------------------------------------

    def test_dimensions_inches_single(self):
        """Test '12-inch' format required by study guide."""
        l, w, h, v = parse_dimensions_cm("Stainless Steel Pizza Pan 12-inch")
        self.assertAlmostEqual(l, 30.48, places=2)
        self.assertIsNone(w)
        self.assertIsNone(h)

        l_quote, _, _, _ = parse_dimensions_cm('MacBook Air 13.3" Screen')
        self.assertAlmostEqual(l_quote, 33.782, places=2)

    def test_dimensions_cm_and_mm(self):
        l, _, _, _ = parse_dimensions_cm("Plastic Ruler 30 cm")
        self.assertAlmostEqual(l, 30.0, places=1)

        l_mm, _, _, _ = parse_dimensions_cm("Derma Roller 1.5mm titanium needles")
        self.assertAlmostEqual(l_mm, 0.15, places=2)

    def test_dimensions_2d_and_3d(self):
        l2, w2, h2, area = parse_dimensions_cm("Yoga Mat 60 x 180 cm")
        self.assertAlmostEqual(l2, 180.0, places=1)
        self.assertAlmostEqual(w2, 60.0, places=1)
        self.assertAlmostEqual(area, 10800.0, places=1)

        l3, w3, h3, vol = parse_dimensions_cm("Storage Box 10 x 20 x 30 cm")
        self.assertAlmostEqual(l3, 30.0, places=1)
        self.assertAlmostEqual(w3, 20.0, places=1)
        self.assertAlmostEqual(h3, 10.0, places=1)
        self.assertAlmostEqual(vol, 6000.0, places=1)

    # -----------------------------------------------------------------------
    # Electrical, Power, Storage Tests
    # -----------------------------------------------------------------------

    def test_power_watts(self):
        """Test '60 W' format required by study guide."""
        val = parse_power_watts("LED Desk Lamp 60 W Energy Saving")
        self.assertAlmostEqual(val, 60.0, places=1)

        val_kw = parse_power_watts("Electric Water Heater 2 kw")
        self.assertAlmostEqual(val_kw, 2000.0, places=1)

    def test_voltage_and_battery(self):
        volt = parse_voltage_volts("Cordless Drill 18 v")
        self.assertAlmostEqual(volt, 18.0, places=1)

        bat = parse_battery_mah("Power Bank 10000 mah")
        self.assertAlmostEqual(bat, 10000.0, places=1)

    def test_storage_gb_and_tb(self):
        gb = parse_storage_gb("High Speed MicroSD 128 GB")
        self.assertAlmostEqual(gb, 128.0, places=1)

        tb = parse_storage_gb("External Backup Drive 2 TB")
        self.assertAlmostEqual(tb, 2048.0, places=1)

        mb = parse_storage_gb("Retro Flash Card 512 mb")
        self.assertAlmostEqual(mb, 0.5, places=2)

    # -----------------------------------------------------------------------
    # Percentage & Model Identifiers Tests
    # -----------------------------------------------------------------------

    def test_percentage_extraction(self):
        pct = parse_percentage("Lee Posh Lactic Acid 60% Peel")
        self.assertAlmostEqual(pct, 60.0, places=1)

    def test_model_identifiers(self):
        """Test model identifiers required by study guide."""
        has_code, count, max_len = extract_model_identifiers("Sony WH-1000XM4 Noise Canceling Headphones")
        self.assertEqual(has_code, 1)
        self.assertGreaterEqual(count, 1)
        self.assertGreaterEqual(max_len, 8)

        has_asin, _, _ = extract_model_identifiers("Catalog Item B072BGHNJ1")
        self.assertEqual(has_asin, 1)

    def test_common_words_not_marked_as_model_codes(self):
        has_code, count, _ = extract_model_identifiers("COMBO PACK SIZE INCH")
        self.assertEqual(has_code, 0)
        self.assertEqual(count, 0)

    # -----------------------------------------------------------------------
    # Null & Empty Robustness
    # -----------------------------------------------------------------------

    def test_null_and_empty_inputs_do_not_crash(self):
        self.assertEqual(parse_pack_count(None), (1.0, 0))

    def test_pandas_na_inputs_do_not_crash(self):
        self.assertIsNone(parse_weight_grams(pd.NA))
        self.assertIsNone(parse_volume_ml(pd.NA))
        self.assertEqual(parse_pack_count(pd.NA), (1.0, 0))

    def test_explicit_total_weight_is_not_multiplied_by_pack(self):
        specs = extract_row_physical_specs(
            "Pack of 6 total net weight 1500 g", None, None
        )
        self.assertEqual(specs["pack_count"], 6.0)
        self.assertEqual(specs["total_weight_g"], 1500.0)
        self.assertEqual(specs["weight_is_explicit_total"], 1)

    def test_explicit_total_without_word_weight_or_volume(self):
        specs_w = extract_row_physical_specs("Protein Powder, Pack of 2, Total 2 kg", None, None)
        self.assertEqual(specs_w["pack_count"], 2.0)
        self.assertEqual(specs_w["total_weight_g"], 2000.0)
        self.assertEqual(specs_w["weight_is_explicit_total"], 1)

        specs_v = extract_row_physical_specs("Red Bull Energy Drink, Pack of 4, Total: 1000ml", None, None)
        self.assertEqual(specs_v["pack_count"], 4.0)
        self.assertEqual(specs_v["total_volume_ml"], 1000.0)
        self.assertEqual(specs_v["volume_is_explicit_total"], 1)

    def test_per_unit_weight_with_net_wt_not_treated_as_explicit_total(self):
        specs_unit = extract_row_physical_specs("Dove Soap Bar, Pack of 4, Net Wt 100g each", None, None)
        self.assertEqual(specs_unit["pack_count"], 4.0)
        self.assertEqual(specs_unit["total_weight_g"], 400.0)
        self.assertEqual(specs_unit["weight_is_explicit_total"], 0)
        self.assertEqual(specs_unit["weight_is_per_item"], 1)

        specs_combo = extract_row_physical_specs("Dove Soap Bar, Pack of 4, Net Wt 400g (100g each)", None, None)
        self.assertEqual(specs_combo["pack_count"], 4.0)
        self.assertEqual(specs_combo["total_weight_g"], 400.0)
        self.assertEqual(specs_combo["weight_is_explicit_total"], 1)

    def test_chunk_extractor_returns_indexed_dataframe(self):
        source = pd.DataFrame(
            {"TITLE": ["Rice 1 kg", "Soap pack of 2 50 g each"]},
            index=[10, 20],
        )
        result = _extract_chunk(source, "TITLE", "PACK_SIZE", "DESCRIPTION")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.index.tolist(), [10, 20])
        self.assertIn("total_weight_g", result.columns)
        self.assertEqual(parse_pack_count(""), (1.0, 0))
        self.assertIsNone(parse_weight_grams(None))
        self.assertIsNone(parse_volume_ml(""))
        self.assertEqual(parse_dimensions_cm(None), (None, None, None, None))
        self.assertIsNone(parse_power_watts(None))
        self.assertIsNone(parse_storage_gb(""))
        self.assertEqual(extract_model_identifiers(None), (0, 0, 0))

    # -----------------------------------------------------------------------
    # Text Surface Statistics Tests
    # -----------------------------------------------------------------------

    def test_compute_text_surface_stats(self):
        s = pd.Series(["Brand New Watch | 40mm - Waterproof", "", None])
        stats = compute_text_surface_stats(s, prefix="title")
        self.assertEqual(len(stats), 3)
        self.assertIn("title_char_len", stats.columns)
        self.assertIn("title_digit_ratio", stats.columns)
        self.assertIn("title_delimiters_count", stats.columns)
        self.assertIn("title_is_empty", stats.columns)
        self.assertEqual(stats["title_is_empty"].tolist(), [0, 1, 1])
        self.assertGreater(stats.loc[0, "title_delimiters_count"], 0)

    # -----------------------------------------------------------------------
    # Fold-Safe Target Encoding Tests
    # -----------------------------------------------------------------------

    def test_fold_safe_target_encoder_zero_leakage(self):
        """Verify that a row's target does NOT influence its own out-of-fold encoding."""
        train_df = pd.DataFrame({
            "brand": ["BrandA", "BrandA", "BrandB", "BrandB", "BrandA", "BrandB"],
            "target": [10.0, 20.0, 100.0, 200.0, 30.0, 300.0],
            "fold": [0, 0, 1, 1, 2, 2],
        })

        encoder = FoldSafeTargetEncoder(
            columns=["brand"],
            target_column="target",
            fold_column="fold",
            m=1.0,
            transform_target="none",
        )
        encoder.fit(train_df)
        oof_encoded = encoder.transform(train_df, is_train_oof=True)

        encoded_f0_row0 = oof_encoded.loc[0, "te_brand"]
        encoded_f0_row1 = oof_encoded.loc[1, "te_brand"]
        self.assertEqual(encoded_f0_row0, encoded_f0_row1)

        modified_train = train_df.copy()
        modified_train.loc[0, "target"] = 10_000.0
        encoder_mod = FoldSafeTargetEncoder(
            columns=["brand"], target_column="target", fold_column="fold", m=1.0, transform_target="none"
        )
        encoder_mod.fit(modified_train)
        oof_mod = encoder_mod.transform(modified_train, is_train_oof=True)
        self.assertEqual(oof_encoded.loc[0, "te_brand"], oof_mod.loc[0, "te_brand"])

    def test_target_encoding_unseen_category_receives_global_mean(self):
        train_df = pd.DataFrame({
            "brand": ["BrandA", "BrandA", "BrandB", "BrandB"],
            "target": [10.0, 20.0, 30.0, 40.0],
            "fold": [0, 1, 0, 1],
        })
        encoder = FoldSafeTargetEncoder(
            columns=["brand"], target_column="target", m=1.0, transform_target="none"
        )
        encoder.fit(train_df)

        test_df = pd.DataFrame({"brand": ["UnseenBrandX"]})
        test_encoded = encoder.transform(test_df, is_train_oof=False)
        self.assertAlmostEqual(test_encoded.loc[0, "te_brand"], float(train_df["target"].mean()), places=3)

    def test_target_encoder_is_oof_when_y_is_passed_separately(self):
        X = pd.DataFrame({
            "brand": ["a", "b", "c", "d"],
            "fold": [0, 1, 0, 1],
        })
        y = pd.Series([10.0, 20.0, 100.0, 200.0])
        encoder = FoldSafeTargetEncoder(
            columns=["brand"], target_column="target", fold_column="fold", m=0.0
        )
        encoded = encoder.fit_transform(X, y=y, is_train_oof=True)["te_brand"].to_numpy()
        self.assertFalse(np.allclose(encoded, np.log1p(y)))
        self.assertTrue(np.isfinite(encoded).all())

    def test_target_encoder_missing_folds_fails_closed(self):
        X = pd.DataFrame({"brand": ["a", "b"], "target": [10.0, 20.0]})
        encoder = FoldSafeTargetEncoder(["brand"], "target", m=1.0).fit(X)
        with self.assertRaises(ValueError):
            encoder.transform(X, is_train_oof=True)

    # -----------------------------------------------------------------------
    # Full Pipeline, Imputation, and Domain Interactions Tests
    # -----------------------------------------------------------------------

    def test_catalog_feature_extractor_pipeline(self):
        train_df = pd.DataFrame({
            "PRODUCT_ID": [101, 102, 103],
            "TITLE": [
                "Lee Posh Lactic Acid 60% Peel 250 ml",
                "Stainless Steel Knife Set of 4 1.5 kg",
                "Sony WH-1000XM4 Wireless Headphones 60 W",
            ],
            "DESCRIPTION": [
                "Anti-ageing serum",
                None,
                "Over-ear noise cancelling",
            ],
            "PACK_SIZE": [None, "4 pcs", None],
            "BRAND": ["Lee Posh", "Generic", "Sony"],
            "CATEGORY": ["Skin Care", "Kitchen", "Electronics"],
            "PRICE": [799.0, 1200.0, 24000.0],
            "fold": [0, 1, 2],
        })

        extractor = CatalogFeatureExtractor(
            title_column="TITLE",
            desc_column="DESCRIPTION",
            pack_column="PACK_SIZE",
            brand_column="BRAND",
            category_column="CATEGORY",
            target_column="PRICE",
            fold_column="fold",
        )

        features = extractor.fit_transform(train_df, is_train_oof=True)

        self.assertEqual(len(features), 3)
        self.assertIn("pack_count", features.columns)
        self.assertIn("unit_volume_ml", features.columns)
        self.assertIn("unit_weight_g", features.columns)
        self.assertIn("power_watts", features.columns)
        self.assertIn("has_model_code", features.columns)
        self.assertIn("is_tech_spec", features.columns)
        self.assertIn("te_BRAND", features.columns)
        self.assertIn("te_CATEGORY", features.columns)

        # Check extracted row 0
        self.assertEqual(features.loc[0, "unit_volume_ml"], 250.0)
        self.assertEqual(features.loc[0, "percentage_spec"], 60.0)

        # Check extracted row 1 (pack of 4, 1.5kg)
        self.assertEqual(features.loc[1, "pack_count"], 4.0)
        self.assertEqual(features.loc[1, "unit_weight_g"], 1500.0)
        self.assertEqual(features.loc[1, "total_weight_g"], 6000.0)

        # Check extracted row 2 (Sony WH-1000XM4, 60 W)
        self.assertEqual(features.loc[2, "has_model_code"], 1)
        self.assertEqual(features.loc[2, "power_watts"], 60.0)
        self.assertEqual(features.loc[2, "is_tech_spec"], 1)

    def test_catalog_feature_extractor_linear_imputation(self):
        """Verify that impute_missing=True eliminates all NaNs for linear models."""
        df = pd.DataFrame({
            "TITLE": ["Headphones 60 W", "Notebook 100 pages"],
            "DESCRIPTION": ["Music", "Stationery"],
            "PACK_SIZE": [None, None],
            "BRAND": ["Sony", "Classmate"],
            "CATEGORY": ["Electronics", "Office"],
        })
        extractor = CatalogFeatureExtractor(impute_missing=True, enable_target_encoding=False)
        features = extractor.fit_transform(df)

        # Must have zero NaNs across all columns
        self.assertEqual(features.isna().sum().sum(), 0)

    def test_catalog_feature_extractor_immutability(self):
        df = pd.DataFrame({
            "TITLE": ["Product 1 500g"],
            "DESCRIPTION": ["Desc 1"],
            "PACK_SIZE": ["1 pack"],
            "BRAND": ["BrandA"],
            "CATEGORY": ["CatA"],
        })
        original = df.copy(deep=True)
        extractor = CatalogFeatureExtractor(enable_target_encoding=False)
        _ = extractor.fit_transform(df)
        pd.testing.assert_frame_equal(df, original)

    def test_catalog_feature_schema_is_stable_when_test_column_is_missing(self):
        train = pd.DataFrame({
            "TITLE": ["a", "b", "c", "d"],
            "BRAND": ["A", "B", "A", "B"],
            "CATEGORY": ["X", "X", "Y", "Y"],
            "PRICE": [10.0, 20.0, 30.0, 40.0],
            "fold": [0, 1, 0, 1],
        })
        test = pd.DataFrame({"TITLE": ["new"], "CATEGORY": ["X"]})
        extractor = CatalogFeatureExtractor(target_column="PRICE")
        train_features = extractor.fit_transform(train, is_train_oof=True)
        test_features = extractor.transform(test)
        self.assertEqual(list(train_features.columns), list(test_features.columns))
        self.assertEqual(test_features.loc[0, "is_missing_brand"], 1)
        self.assertTrue(np.isfinite(test_features.loc[0, "te_BRAND"]))


if __name__ == "__main__":
    unittest.main()
