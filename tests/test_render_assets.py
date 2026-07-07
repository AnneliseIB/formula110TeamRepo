from __future__ import annotations

from dataclasses import fields

from racing.game.app import DEFAULT_WINDOW_ICON_PATH
from racing.graphics.render_assets import SceneAssets, material_specular_strength


def test_scene_assets_only_contains_current_formula_rendering_fields() -> None:
    field_names = {field.name for field in fields(SceneAssets)}

    assert "team_paint_material" in field_names
    assert "tire_texture" in field_names
    assert "argyle_banner_texture" in field_names
    assert "argyle_banner_material" in field_names
    assert "formula_banner_texture" in field_names
    assert "formula_banner_material" in field_names
    assert "silver_body_texture" not in field_names
    assert "lime_body_material" not in field_names


def test_material_specular_strength_decreases_with_roughness() -> None:
    glossy = material_specular_strength(roughness=0.1, metallic=0.0)
    matte = material_specular_strength(roughness=0.8, metallic=0.0)

    assert glossy > matte
    assert matte >= 0.02


def test_default_window_icon_asset_is_packaged_ico() -> None:
    icon_bytes = DEFAULT_WINDOW_ICON_PATH.read_bytes()

    assert DEFAULT_WINDOW_ICON_PATH.name == "ursina.ico"
    assert icon_bytes[:4] == b"\x00\x00\x01\x00"
    assert int.from_bytes(icon_bytes[4:6], byteorder="little") >= 1
