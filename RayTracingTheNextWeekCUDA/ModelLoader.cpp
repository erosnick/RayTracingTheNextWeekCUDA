#include "ModelLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

struct SimpleVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texcoord;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 tangent;
    glm::vec3 binormal;
    glm::vec3 normal;
    glm::vec2 texCoord;

    bool operator==(const Vertex& other) const {
        return (other.position == position) &&
            (other.normal == normal) &&
            (other.texCoord == texCoord);
    }
};

//namespace std {
//    template<> struct hash<Vertex> {
//        size_t operator()(Vertex const& vertex) const {
//            return (hash<glm::vec3>()(vertex.position))
//                 ^ (hash<glm::vec3>()(vertex.normal) << 1)
//                 ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
//        }
//    };
//}

class Mesh {
public:
    Mesh() {

    }

    ~Mesh() {
    }

    void addVertex(const Vertex& vertex) {
        vertices.push_back(vertex);
        vertexCount = vertices.size();
    }

    void addIndex(uint32_t index) {
        indices.push_back(index);
    }

    size_t getVertexBufferByteSize() const {
        return sizeof(Vertex) * vertices.size();
    }

    size_t getIndexBufferByteSize() const {
        return sizeof(uint32_t) * indices.size();
    }

    uint32_t getTriangleCount() const {
        return static_cast<uint32_t>(vertices.size() / 3);
    }

    uint32_t getVertexCount() const {
        return vertexCount;
    }

    void setName(const std::string& inName) {
        name = inName;
    }

    const std::string& getName() const {
        return name;
    }

    std::vector<Vertex> getVertices() const {
        return vertices;
    }

    const Vertex* getVerticesData() const {
        return vertices.data();
    }

    std::vector<uint32_t> getIndices() const {
        return indices;
    }

    int32_t getIndexCount() const {
        return static_cast<int32_t>(indices.size());
    }

    const uint32_t* getIndicesData() const {
        return indices.data();
    }

    int32_t getNormalIndexCount() const {
        return static_cast<int32_t>(normals.size() * 2);
    }

    void computeTangentSpace();

private:
    std::string name;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SimpleVertex> normals;

    uint32_t vertexCount = 0;
};

class Model {
public:
    Model() {
        position = glm::vec3(0.0f);
        transform = glm::mat4(1.0f);
    }

    ~Model() {
    }

    uint32_t getTriangleCount() const {
        return 0;
    }

    void scale(const glm::vec3& factor) {
        transform = glm::scale(transform, factor);
    }

    void translate(const glm::vec3& offset) {
        transform = glm::translate(transform, offset);
    }

    void rotate(float angle, const glm::vec3& axis) {
        transform = glm::rotate(transform, glm::radians(angle), axis);
    }

    void setPosition(const glm::vec3& inPosition) {
        position = inPosition;
        transform[3][0] = inPosition.x;
        transform[3][1] = inPosition.y;
        transform[3][2] = inPosition.z;
    }

    glm::vec3 getPosition() const {
        return position;
    }

    void setTransform(const glm::mat4& inTransform) {
        transform = inTransform;
    }

    glm::mat4 getTransform() const {
        return transform;
    }

    void computeTangentSpace();

    void setName(const std::string& inName) {
        name = inName;
    }

    const std::string& getName() const {
        return name;
    }

    void addMesh(const std::shared_ptr<Mesh>& mesh) {
        meshes.push_back(mesh);
        triangleCount += mesh->getTriangleCount();
    }

    const std::vector<std::shared_ptr<Mesh>>& getMeshes() const {
        return meshes;
    }

    size_t getMeshCount() const {
        return meshes.size();
    }
private:
    std::string name;
    glm::vec3 position;
    glm::mat4 transform;
    std::vector <std::shared_ptr<Mesh>> meshes;

    uint32_t triangleCount = 0;
};

void Model::computeTangentSpace() {
    for (auto& mesh : meshes) {
        mesh->computeTangentSpace();
    }
}

void Mesh::computeTangentSpace() {
    std::vector<glm::vec3 > tangent;
    std::vector<glm::vec3 > binormal;

    for (unsigned int i = 0; i < getIndexCount(); i += 3) {

        glm::vec3 vertex0 = vertices.at(indices.at(i + 0)).position;
        glm::vec3 vertex1 = vertices.at(indices.at(i + 1)).position;
        glm::vec3 vertex2 = vertices.at(indices.at(i + 2)).position;

        glm::vec3 normal0 = vertices.at(indices.at(i + 0)).normal;
        glm::vec3 normal1 = vertices.at(indices.at(i + 1)).normal;
        glm::vec3 normal2 = vertices.at(indices.at(i + 2)).normal;

        glm::vec3 normal = glm::normalize(glm::cross((vertex1 - vertex0), (vertex2 - vertex0)));

        normal = (normal0 + normal1 + normal2) / (glm::length(normal0 + normal1 + normal2));

        glm::vec3 deltaPos;
        if (vertex0 == vertex1)
            deltaPos = vertex2 - vertex0;
        else
            deltaPos = vertex1 - vertex0;

        glm::vec2 uv0 = vertices.at(indices.at(i + 0)).texCoord;
        glm::vec2 uv1 = vertices.at(indices.at(i + 1)).texCoord;
        glm::vec2 uv2 = vertices.at(indices.at(i + 2)).texCoord;

        glm::vec2 deltaUV1 = uv1 - uv0;
        glm::vec2 deltaUV2 = uv2 - uv0;

        glm::vec3 tangent; // tangents
        glm::vec3 binormal; // binormal

        // avoid division with 0
        if (deltaUV1.s != 0)
            tangent = deltaPos / deltaUV1.s;
        else
            tangent = deltaPos / 1.0f;

        tangent = glm::normalize(tangent - glm::dot(normal, tangent) * normal);

        binormal = glm::normalize(glm::cross(tangent, normal));

        //vertices[indices.at(i + 0)].normal = { normal.x, normal.y, normal.z };
        //vertices[indices.at(i + 1)].normal = { normal.x, normal.y, normal.z };
        //vertices[indices.at(i + 2)].normal = { normal.x, normal.y, normal.z };

        // write into array - for each vertex of the face the same value
        vertices[indices.at(i + 0)].tangent = { tangent.x, tangent.y, tangent.z };
        vertices[indices.at(i + 1)].tangent = { tangent.x, tangent.y, tangent.z };
        vertices[indices.at(i + 2)].tangent = { tangent.x, tangent.y, tangent.z };

        vertices[indices.at(i + 0)].binormal = { binormal.x, binormal.y, binormal.z };
        vertices[indices.at(i + 1)].binormal = { binormal.x, binormal.y, binormal.z };
        vertices[indices.at(i + 2)].binormal = { binormal.x, binormal.y, binormal.z };
    }
}

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.position)
                ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1)
                ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

std::shared_ptr<Model> loadModel(const std::string& fileName, const std::string& inName = "Model", const std::string& materialPath = ".") {
    tinyobj::ObjReaderConfig readConfig;
    readConfig.mtl_search_path = materialPath;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(fileName, readConfig)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjRead: " << reader.Error();
        }
        return nullptr;
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto slash = fileName.find_last_of('/');
    auto dot = fileName.find_last_of('.');

    auto model = std::make_shared<Model>();

    if (!inName.empty()) {
        model->setName(inName);
    }
    else {
        model->setName(fileName.substr(slash + 1, dot - (slash + 1)));
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // 三层循环分别遍历mesh(s), face(s), vertex(s)
    // 不进行任何过滤的情况下，获得的顶点数组会包含
    // 荣誉数据(比如支持贴图和光照的立方体最多只需24个顶点)
    // 而这里会产生36个顶点，它们的排布是线性的，绘制的时候
    // 直接调用glDrawArrays即可，但是更优化的做法是过滤掉
    // 冗余的顶点，并使用glDrawElements来进行绘制
    // Loop over shapes
    //for (size_t s = 0; s < shapes.size(); s++) {
    //    // Loop over faces(polygon)
    //    size_t indexOffset = 0;
    //    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
    //        size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

    //        // Loop over vertices in the face
    //        for (size_t v = 0; v < fv; v++) {
    //            // Access to vertex
    //            tinyobj::index_t index = shapes[s].mesh.indices[indexOffset + v];
    //            tinyobj::real_t vx = attrib.vertices[3 * size_t(index.vertex_index) + 0];
    //            tinyobj::real_t vy = attrib.vertices[3 * size_t(index.vertex_index) + 1];
    //            tinyobj::real_t vz = attrib.vertices[3 * size_t(index.vertex_index) + 2];

    //            Vertex vertex;

    //            vertex.position = { vx, vy, vz };

    //            // Check if 'normal_index' is zero of positive. negative = no normal data
    //            if (index.normal_index >= 0) {
    //                tinyobj::real_t nx = attrib.normals[3 * size_t(index.normal_index) + 0];
    //                tinyobj::real_t ny = attrib.normals[3 * size_t(index.normal_index) + 1];
//                tinyobj::real_t nz = attrib.normals[3 * size_t(index.normal_index) + 2];
//                vertex.normal = { nx, ny, nz };
//            }

//            // Check if 'texcoord_index' is zero or positive. negative = no texcoord data
//            if (index.texcoord_index >= 0) {
//                tinyobj::real_t tx = attrib.texcoords[2 * size_t(index.texcoord_index) + 0];
//                tinyobj::real_t ty = 1.0f - attrib.texcoords[2 * size_t(index.texcoord_index) + 1];
//                vertex.texCoord = { tx, ty };
//            }

//            // 只有顶点的每个分量都相同时(这里包含position, normal和texcoord)，才考虑它是唯一
//            // 举例来说，要让一个立方体能支持正确的纹理贴图和光照计算，总计需要24个顶点
//            // 因为立方体有6个面，每个面4个顶点，总计24个顶点，都需要有独立的normal和texcoord
//            if (uniqueVertices.count(vertex) == 0) {
//                uniqueVertices[vertex] = static_cast<uint32_t>(model.getVertices().size());
//                model.addVertex(vertex);
//            }

//            // 如函数开头所说，tinyobjloader读取出的顶点数据是线性的
//            // 还是以立方体为例，36个顶点数组对应的索引数组就是0~35
//            // 上面经过顶点去重之后，索引也要跟着更新，既然顶点数组
//            // 是线性的，我们可以直接用去重过后的顶点数组尺寸做索引
//            // 所有的索引值肯定是位于0 ~ model.vertices.size() - 1
//            // 之间。model.indices数组的大小最后一定是
//            // shapes[s].mesh.num_face_vertices.size() * fv，
//            // 也就是三角形面数 * 3
//            model.addIndex(uniqueVertices[vertex]);
//        }
//        indexOffset += fv;

//        // per-face material
//        shapes[s].mesh.material_ids[f];
//    }
//} 
    size_t materialIndex = 0;
    uint32_t counter = 0;
    for (const auto& shape : shapes) {
        auto mesh = std::make_shared<Mesh>();
        std::unordered_map<Vertex, uint32_t> uniqueVertices;
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex = {};

            vertex.position = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            // Check if 'normal_index' is zero of positive. negative = no normal data
            if (index.normal_index >= 0) {
                tinyobj::real_t nx = attrib.normals[3 * size_t(index.normal_index) + 0];
                tinyobj::real_t ny = attrib.normals[3 * size_t(index.normal_index) + 1];
                tinyobj::real_t nz = attrib.normals[3 * size_t(index.normal_index) + 2];
                vertex.normal = { nx, ny, nz };
            }

            if (index.texcoord_index >= 0) {
                vertex.texCoord = {
                  attrib.texcoords[2 * index.texcoord_index + 0],
                  1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };
            }

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(mesh->getVertexCount());
                mesh->addVertex(vertex);
            }

            mesh->addIndex(uniqueVertices[vertex]);

        }

        mesh->setName(shape.name);

        model->addMesh(std::move(mesh));
    }

    return model;
}

std::vector<Float3> loadModel(const std::string& fileName) {
    auto model = loadModel(fileName, "model");

    auto mesh = model->getMeshes()[0];

    auto vertices = mesh->getVertices();
    auto indices = mesh->getIndices();

    std::vector<Float3> resultVertices;

    for (auto index : indices) {
        const auto& vertex = vertices[index];
        Float3 resultVertex = { vertex.position.x, vertex.position.y, vertex.position.z };
        resultVertices.push_back(resultVertex);
    }

    return resultVertices;
}
