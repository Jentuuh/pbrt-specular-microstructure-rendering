#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_GLITTER_H
#define PBRT_MATERIALS_GLITTER_H

#include "pbrt.h"
#include "material.h"
#include "spectrum.h"

namespace pbrt {
    class GlitterMaterial : public Material
    {
    public:
        GlitterMaterial(const std::shared_ptr<Texture<Spectrum>>& eta,
            const std::shared_ptr<Texture<Spectrum>>& k,
            const std::shared_ptr<Texture<Float>>& rough,
            const std::shared_ptr<Texture<Float>>& urough,
            const std::shared_ptr<Texture<Float>>& vrough,
            const std::shared_ptr<Texture<Float>>& bump,
            bool remapRoughness);

        void ComputeScatteringFunctions(SurfaceInteraction* si, MemoryArena& arena,
            TransportMode mode,
            bool allowMultipleLobes) const;

    private:
        std::shared_ptr<Texture<Spectrum>> eta, k;
        std::shared_ptr<Texture<Float>> roughness, uRoughness, vRoughness;
        std::shared_ptr<Texture<Float>> bumpMap;
        bool remapRoughness;
    };

    GlitterMaterial* CreateGlitterMaterial (const TextureParams& mp);

}

#endif  // PBRT_MATERIALS_METAL_H


