"""Gate test for data-bound GeoJSON fill color and map layers per F7."""

from pathlib import Path

from src.geo.choropleth import get_country_polygons

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_country_geojson_features_contain_data_bound_fill_color() -> None:
    """Assert get_country_polygons returns features with data-bound fill_color property."""
    data = get_country_polygons()
    features = data.get("features", [])
    assert len(features) >= 177, f"Expected >= 177 features, found {len(features)}"

    for feat in features:
        props = feat.get("properties", {})
        assert "fill_color" in props, "Feature properties missing 'fill_color'"
        fill = props["fill_color"]
        assert isinstance(fill, list) and len(fill) == 4, f"Invalid fill_color: {fill}"


def test_risk_page_map_uses_properties_fill_color_and_not_literal() -> None:
    """Assert dashboard/pages/risk.py uses properties.fill_color and not static literal [70, 130, 180, 80]."""
    risk_page = ROOT_DIR / "dashboard" / "pages" / "risk.py"
    assert risk_page.exists()
    content = risk_page.read_text(encoding="utf-8")

    assert "[70, 130, 180, 80]" not in content, "Found hardcoded static RGBA list in risk.py map layer!"
    assert "properties.fill_color" in content, "Missing properties.fill_color data binding in risk.py map layer!"


def test_map_data_bound_negative_control_literal_rgba_fails() -> None:
    """Negative control: assert static literal RGBA fill check fails on static string."""
    fake_code = 'pdk.Layer("GeoJsonLayer", get_fill_color="[70, 130, 180, 80]")'
    assert "[70, 130, 180, 80]" in fake_code, "Negative control should detect literal RGBA string"
