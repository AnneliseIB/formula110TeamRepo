"""Procedural textures and materials used by the rendered scenes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module, resources
from math import pi, sin
from typing import Any, cast

from racing.track.world import clamp

CONCRETE_BASE_COLOR = (0.255, 0.255, 0.245, 1)
INNER_WALL_LIGHTNESS_SCALE = 1.80
INNER_WALL_CONCRETE_COLOR = (
    CONCRETE_BASE_COLOR[0] * INNER_WALL_LIGHTNESS_SCALE,
    CONCRETE_BASE_COLOR[1] * INNER_WALL_LIGHTNESS_SCALE,
    CONCRETE_BASE_COLOR[2] * INNER_WALL_LIGHTNESS_SCALE,
    1,
)
CONCRETE_TEXTURE_BASE_SHADE = 0.27
CONCRETE_ROUGHNESS = 0.60
INNER_WALL_REFLECTANCE_SCALE = 1.20
WALL_CONCRETE_DARKNESS_SCALE = 0.75
TEAM_PAINT_ROUGHNESS = 0.175
TEAM_PAINT_REFLECTANCE_SCALE = 0.50
PREVIOUS_TEAM_PAINT_ROUGHNESS = 0.045
CARBON_REFLECTANCE_SCALE = 1.25
CAROLINA_BLUE_COLOR = (75 / 255, 156 / 255, 211 / 255, 1.0)
WALL_PAINT_ROUGHNESS = 0.035
ARGYLE_BANNER_TEXTURE_PATH = ("assets", "graphics", "argyle.png")
FORMULA_BANNER_TEXTURE_PATH = ("assets", "graphics", "formula110.png")


@dataclass(frozen=True, slots=True)
class SceneAssets:
    """Procedural textures and materials shared by scene builders."""

    asphalt_texture: Any
    curb_texture: Any
    wall_texture: Any
    grass_texture: Any
    gravel_texture: Any
    tire_texture: Any
    argyle_banner_texture: Any
    formula_banner_texture: Any
    asphalt_material: Any
    concrete_material: Any
    inner_wall_material: Any
    kerb_material: Any
    black_plastic_material: Any
    glass_material: Any
    rubber_material: Any
    argyle_banner_material: Any
    formula_banner_material: Any
    team_paint_material: Any
    wall_paint_material: Any
    white_decal_material: Any
    red_decal_material: Any
    yellow_decal_material: Any


def lit_entity(ursina: Any, **kwargs: Any) -> Any:
    """Create an Ursina entity and attach a material when supplied."""
    material = kwargs.pop("material", None)
    entity = ursina.Entity(**kwargs)
    if material is not None:
        entity.setMaterial(material, 1)
    return entity


def create_scene_assets() -> SceneAssets:
    """Create procedural textures and materials for scene rendering."""
    return SceneAssets(
        asphalt_texture=_texture_from_pixels("procedural-asphalt", 256, asphalt_pixel),
        curb_texture=_texture_from_pixels("procedural-curb", 128, _curb_pixel),
        wall_texture=_texture_from_pixels("procedural-concrete", 256, _concrete_pixel),
        grass_texture=_texture_from_pixels("procedural-grass", 256, _grass_pixel),
        gravel_texture=_texture_from_pixels("procedural-gravel", 192, _gravel_pixel),
        tire_texture=_texture_from_pixels("procedural-rubber", 128, _rubber_pixel),
        argyle_banner_texture=_white_backed_texture_from_resource_png(
            "unc-argyle-banner",
            relative_path=ARGYLE_BANNER_TEXTURE_PATH,
            repeat_u=True,
            repeat_v=False,
        ),
        formula_banner_texture=_white_backed_texture_from_resource_png(
            "formula-110-banner",
            relative_path=FORMULA_BANNER_TEXTURE_PATH,
            repeat_u=False,
            repeat_v=False,
        ),
        asphalt_material=_material("night-asphalt", (0.125, 0.132, 0.144, 1), roughness=0.80, metallic=0.0),
        concrete_material=_material("night-concrete", CONCRETE_BASE_COLOR, roughness=CONCRETE_ROUGHNESS, metallic=0.0),
        inner_wall_material=_material(
            "lightened-inner-wall-concrete",
            INNER_WALL_CONCRETE_COLOR,
            roughness=CONCRETE_ROUGHNESS,
            metallic=0.0,
            specular_scale=INNER_WALL_REFLECTANCE_SCALE,
        ),
        kerb_material=_material("gloss-painted-kerb", (0.94, 0.92, 0.86, 1), roughness=0.22, metallic=0.0),
        black_plastic_material=_material(
            "black-plastic",
            (0.02, 0.023, 0.026, 1),
            roughness=0.48,
            metallic=0.0,
            specular_scale=CARBON_REFLECTANCE_SCALE,
        ),
        glass_material=_material(
            "smoked-polycarbonate-windshield", (0.015, 0.018, 0.022, 1), roughness=0.08, metallic=0.0
        ),
        rubber_material=_material("soft-black-rubber", (0.025, 0.025, 0.024, 1), roughness=0.92, metallic=0.0),
        argyle_banner_material=_material(
            "white-backed-argyle-banner",
            (1.0, 1.0, 1.0, 1),
            roughness=0.24,
            metallic=0.0,
        ),
        formula_banner_material=_material(
            "white-backed-formula-banner",
            (1.0, 1.0, 1.0, 1),
            roughness=0.24,
            metallic=0.0,
        ),
        team_paint_material=_material(
            "gloss-team-paint",
            (0.98, 0.96, 0.92, 1),
            roughness=TEAM_PAINT_ROUGHNESS,
            metallic=0.0,
            specular_scale=TEAM_PAINT_REFLECTANCE_SCALE,
        ),
        wall_paint_material=_material(
            "neon-reflective-carolina-blue-wall-paint",
            (0.98, 1.0, 1.0, 1),
            roughness=WALL_PAINT_ROUGHNESS,
            metallic=0.0,
        ),
        white_decal_material=_material("warm-white-vinyl-decal", (0.94, 0.91, 0.82, 1), roughness=0.34, metallic=0.0),
        red_decal_material=_material("red-vinyl-decal", (0.88, 0.03, 0.02, 1), roughness=0.36, metallic=0.0),
        yellow_decal_material=_material("yellow-vinyl-lettering", (0.96, 0.95, 0.12, 1), roughness=0.34, metallic=0.0),
    )


def _material(
    name: str,
    base_color: tuple[float, float, float, float],
    *,
    roughness: float,
    metallic: float,
    specular_scale: float = 1.0,
) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    material = core.Material(name)
    specular_strength = material_specular_strength(roughness=roughness, metallic=metallic)
    material.setBaseColor(core.VBase4(*base_color))
    material.setAmbient(core.VBase4(*base_color))
    material.setDiffuse(core.VBase4(*base_color))
    material.setSpecular(
        core.VBase4(
            0.8 * specular_strength * specular_scale,
            0.8 * specular_strength * specular_scale,
            0.75 * specular_strength * specular_scale,
            1,
        )
    )
    material.setShininess(max(1.0, 96.0 * (1.0 - roughness)))
    material.setRoughness(roughness)
    material.setMetallic(metallic)
    return material


def material_specular_strength(*, roughness: float, metallic: float) -> float:
    """Estimate how shiny a material should look from roughness and metalness."""
    return max(0.02, (1.0 - roughness) ** 1.5) * (1.0 + metallic * 0.4)


def _texture_from_pixels(
    name: str,
    size: int,
    pixel: Callable[[float, float, int, int], tuple[float, float, float]],
) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    image = core.PNMImage(size, size)
    for y in range(size):
        v = y / size
        for x in range(size):
            u = x / size
            red, green, blue = _clamped_rgb(pixel(u, v, x, y))
            image.setXel(x, y, red, green, blue)

    texture = core.Texture(name)
    texture.load(image)
    texture.setWrapU(core.Texture.WMRepeat)
    texture.setWrapV(core.Texture.WMRepeat)
    return _wrap_panda_texture(texture)


def _white_backed_texture_from_resource_png(
    name: str,
    *,
    relative_path: tuple[str, ...],
    repeat_u: bool,
    repeat_v: bool,
) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    source = core.PNMImage()
    resource = resources.files("racing")
    for part in relative_path:
        resource = resource.joinpath(part)
    with resources.as_file(resource) as path:
        loaded = bool(source.read(core.Filename.fromOsSpecific(str(path))))
    if not loaded:
        raise RuntimeError(f"could not load texture resource: {'/'.join(relative_path)}")

    image = _image_composited_over_white(core=core, source=source)
    texture = core.Texture(name)
    texture.load(image)
    texture.setWrapU(core.Texture.WMRepeat if repeat_u else core.Texture.WMClamp)
    texture.setWrapV(core.Texture.WMRepeat if repeat_v else core.Texture.WMClamp)
    return _wrap_panda_texture(texture)


def _image_composited_over_white(*, core: Any, source: Any) -> Any:
    width = int(source.getXSize())
    height = int(source.getYSize())
    image = core.PNMImage(width, height)
    has_alpha = bool(source.hasAlpha())
    for y in range(height):
        for x in range(width):
            color = source.getXel(x, y)
            alpha = float(source.getAlpha(x, y)) if has_alpha else 1.0
            image.setXel(
                x,
                y,
                _composite_channel_over_white(float(color[0]), alpha),
                _composite_channel_over_white(float(color[1]), alpha),
                _composite_channel_over_white(float(color[2]), alpha),
            )
    return image


def _composite_channel_over_white(channel: float, alpha: float) -> float:
    return clamp(channel * alpha + (1.0 - alpha), 0.0, 1.0)


def _wrap_panda_texture(texture: Any) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    texture.setMinfilter(core.Texture.FTLinearMipmapLinear)
    texture.setMagfilter(core.Texture.FTLinear)
    texture_class = cast(Any, import_module("ursina.texture")).Texture
    wrapped_texture = texture_class(texture, filtering="mipmap")
    wrapped_texture._cached_image = None
    wrapped_texture.path = None
    return wrapped_texture


def asphalt_pixel(u: float, v: float, x: int, y: int) -> tuple[float, float, float]:
    """Generate one noisy asphalt texture pixel."""
    aggregate = (_value_noise(x // 2, y // 2, 17) - 0.5) * 0.055
    coarse_stone = (_value_noise(x // 5, y // 5, 23) - 0.5) * 0.035
    fine_grain = (_value_noise(x * 5, y * 5, 29) - 0.5) * 0.080
    pitting = -0.060 if _value_noise(x * 2, y * 2, 31) < 0.030 else 0.0
    mineral = 0.105 if _value_noise(x * 3, y * 3, 91) > 0.955 else 0.0
    lane_wear = -0.018 if 0.34 < (u % 1.0) < 0.66 else 0.0
    tar_variation = 0.014 * sin((u * 13.0 + v * 17.0) * pi)
    hairline_crack = -0.040 if abs(((u * 9.0 + sin(v * 19.0) * 0.025) % 1.0) - 0.5) < 0.004 else 0.0
    shade = (
        0.100 + aggregate + coarse_stone + fine_grain + pitting + mineral + lane_wear + tar_variation + hairline_crack
    )
    return (shade * 0.91, shade * 0.93, shade * 0.96)


def _curb_pixel(u: float, v: float, x: int, y: int) -> tuple[float, float, float]:
    scuff = (_value_noise(x // 2, y // 2, 31) - 0.5) * 0.075
    brush = 0.018 * sin((u * 7.0 + v * 18.0) * pi)
    rubber_mark = -0.10 if _value_noise(x, y // 3, 37) > 0.975 else 0.0
    shade = 0.92 + scuff + brush + rubber_mark
    return (shade, shade * 0.985, shade * 0.94)


def _concrete_pixel(u: float, v: float, x: int, y: int) -> tuple[float, float, float]:
    pitting = (_value_noise(x, y, 43) - 0.5) * 0.10
    stain = -0.06 * max(0.0, sin(u * pi * 11 + _value_noise(x // 8, y // 8, 5) * 2.0))
    edge_dirt = -0.08 * max(0.0, 1.0 - v * 5.5)
    quartz = concrete_quartz_fleck(x, y)
    shade = CONCRETE_TEXTURE_BASE_SHADE + pitting + stain + edge_dirt
    return (shade * 0.98 + quartz * 0.92, shade + quartz * 0.96, shade * 1.03 + quartz)


def concrete_quartz_fleck(x: int, y: int) -> float:
    """Return a small bright fleck amount for procedural concrete."""
    quartz_seed = _value_noise(x * 3, y * 3, 149)
    return 0.055 * ((quartz_seed - 0.992) / 0.008) if quartz_seed > 0.992 else 0.0


def _grass_pixel(u: float, v: float, x: int, y: int) -> tuple[float, float, float]:
    blade = 0.030 * sin((u * 90.0 + _value_noise(x // 3, y // 3, 71) * 8.0) * pi)
    patch = (_value_noise(x // 9, y // 9, 73) - 0.5) * 0.055
    fine = (_value_noise(x, y, 79) - 0.5) * 0.035
    green = 0.105 + blade + patch + fine
    return (0.025 + patch * 0.16, green, 0.040 + fine * 0.25)


def _gravel_pixel(u: float, v: float, x: int, y: int) -> tuple[float, float, float]:
    pebble = _value_noise(x, y, 83)
    clump = _value_noise(x // 3, y // 3, 89)
    shade = 0.12 + pebble * 0.08 + clump * 0.06
    warm = 0.018 * sin((u + v) * pi * 18.0)
    return (shade + warm, shade * 0.95, shade * 0.88)


def _rubber_pixel(u: float, v: float, x: int, y: int) -> tuple[float, float, float]:
    rib = 0.025 * sin(u * pi * 56.0)
    scuff = (_value_noise(x, y, 97) - 0.5) * 0.04
    shade = 0.055 + rib + scuff
    return (shade, shade, shade * 0.98)


def _value_noise(x: int, y: int, seed: int) -> float:
    value = (x * 374761393 + y * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    value = (value ^ (value >> 13)) * 1274126177
    return ((value ^ (value >> 16)) & 0xFFFF) / 0xFFFF


def _clamped_rgb(color: tuple[float, float, float]) -> tuple[float, float, float]:
    red, green, blue = color
    return clamp(red, 0.0, 1.0), clamp(green, 0.0, 1.0), clamp(blue, 0.0, 1.0)
