#pragma once

#include <yaml-cpp/yaml.h>

#include "CUDATypes.h"
#include "Sphere.h"
#include "Material.h"

namespace YAML {
    template<>
    struct convert<Float3> {
        static Node encode(const Float3& rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node& node, Float3& rhs) {
            if (!node.IsSequence() || node.size() != 3) {
                return false;
            }

            rhs.x = node[0].as<Float>();
            rhs.y = node[1].as<Float>();
            rhs.z = node[2].as<Float>();
            return true;
        }
    };

    template<>
    struct convert<Sphere> {
        static Node encode(const Sphere& rhs) {
            Node node;
            node["center"].push_back(rhs.center.x);
            node["center"].push_back(rhs.center.y);
            node["center"].push_back(rhs.center.y);
            node["radius"] = rhs.radius;
            node["type"] = (uint32_t)PrimitiveType::Sphere;

            return node;
        }

        static bool decode(const Node& node, Sphere& rhs) {
            //if (!node.IsMap() || node.size() != 3) {
            //    return false;
            //}
            printf("%d\n", node.size());

            rhs.center.x = node["center"][0].as<Float>();
            rhs.center.y = node["center"][1].as<Float>();
            rhs.center.z = node["center"][2].as<Float>();
            rhs.radius = node["radius"].as<Float>();
            rhs.type = static_cast<PrimitiveType>(node["type"].as<uint8_t>());

            return true;
        }
    };

    template<>
    struct convert<Lambertian> {
        static Node encode(const Lambertian& rhs) {
            Node node;
            node["albedo"] = rhs.albedo;
            node["type"] = (uint8_t)MaterialType::Lambertian;

            return node;
        }

        static bool decode(const Node& node, Lambertian& rhs) {
            //if (!node.IsMap() || node.size() != 3) {
            //    return false;
            //}
            printf("%d\n", node.size());

            rhs.albedo = node["albedo"].as<Float3>();
            rhs.type = static_cast<MaterialType>(node["type"].as<uint8_t>());

            return true;
        }
    };
}
