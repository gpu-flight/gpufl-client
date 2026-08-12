// A CPU-only ray-tracing example.  It writes a PPM image containing colored
// spheres (circles on the image plane), their shadows, and floor reflections.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr float kRayBias = 0.001F;
constexpr int kSamplesPerPixel = 4;
constexpr int kMaxBounces = 2;

struct Vec3 {
    float x{};
    float y{};
    float z{};

    Vec3() = default;
    constexpr Vec3(float xValue, float yValue, float zValue)
        : x(xValue), y(yValue), z(zValue) {}

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

constexpr Vec3 operator+(Vec3 a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

constexpr Vec3 operator-(Vec3 a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

constexpr Vec3 operator-(const Vec3& value) {
    return {-value.x, -value.y, -value.z};
}

constexpr Vec3 operator*(const Vec3& value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

constexpr Vec3 operator*(float scale, const Vec3& value) {
    return value * scale;
}

constexpr Vec3 operator*(const Vec3& a, const Vec3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

constexpr Vec3 operator/(const Vec3& value, float divisor) {
    return {value.x / divisor, value.y / divisor, value.z / divisor};
}

constexpr float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 normalize(const Vec3& value) {
    const float length = std::sqrt(dot(value, value));
    return value / length;
}

constexpr Vec3 reflect(const Vec3& incoming, const Vec3& normal) {
    return incoming - normal * (2.0F * dot(incoming, normal));
}

constexpr float clamp(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

struct Ray {
    Vec3 origin;
    Vec3 direction;

    constexpr Vec3 at(float distance) const {
        return origin + direction * distance;
    }
};

struct Material {
    Vec3 albedo;
    float reflectivity{};
};

struct Sphere {
    Vec3 center;
    float radius;
    Material material;
};

struct Plane {
    Vec3 point;
    Vec3 normal;
    Material material;
};

struct Hit {
    float distance{kInfinity};
    Vec3 position;
    Vec3 normal;
    Material material;
    bool isFloor{};
};

struct Scene {
    std::vector<Sphere> spheres;
    Plane floor;
    Vec3 lightPosition;
};

bool intersectSphere(const Ray& ray, const Sphere& sphere, float minDistance,
                     float maxDistance, Hit& hit) {
    const Vec3 centerToRay = ray.origin - sphere.center;
    const float a = dot(ray.direction, ray.direction);
    const float halfB = dot(centerToRay, ray.direction);
    const float c = dot(centerToRay, centerToRay) - sphere.radius * sphere.radius;
    const float discriminant = halfB * halfB - a * c;

    if (discriminant < 0.0F) {
        return false;
    }

    const float squareRoot = std::sqrt(discriminant);
    float distance = (-halfB - squareRoot) / a;
    if (distance <= minDistance || distance >= maxDistance) {
        distance = (-halfB + squareRoot) / a;
        if (distance <= minDistance || distance >= maxDistance) {
            return false;
        }
    }

    hit.distance = distance;
    hit.position = ray.at(distance);
    hit.normal = normalize(hit.position - sphere.center);
    hit.material = sphere.material;
    hit.isFloor = false;
    return true;
}

bool intersectPlane(const Ray& ray, const Plane& plane, float minDistance,
                    float maxDistance, Hit& hit) {
    const float denominator = dot(plane.normal, ray.direction);
    if (std::abs(denominator) < 0.0001F) {
        return false;
    }

    const float distance = dot(plane.point - ray.origin, plane.normal) / denominator;
    if (distance <= minDistance || distance >= maxDistance) {
        return false;
    }

    hit.distance = distance;
    hit.position = ray.at(distance);
    hit.normal = denominator < 0.0F ? plane.normal : -plane.normal;
    hit.material = plane.material;
    hit.isFloor = true;
    return true;
}

bool intersectScene(const Ray& ray, const Scene& scene, float minDistance,
                    float maxDistance, Hit& closestHit) {
    bool foundHit = false;
    float closestDistance = maxDistance;
    Hit candidate;

    for (const Sphere& sphere : scene.spheres) {
        if (intersectSphere(ray, sphere, minDistance, closestDistance, candidate)) {
            foundHit = true;
            closestDistance = candidate.distance;
            closestHit = candidate;
        }
    }

    if (intersectPlane(ray, scene.floor, minDistance, closestDistance, candidate)) {
        foundHit = true;
        closestHit = candidate;
    }

    return foundHit;
}

Vec3 skyColor(const Ray& ray) {
    const float horizon = 0.5F * (ray.direction.y + 1.0F);
    return (1.0F - horizon) * Vec3{0.02F, 0.03F, 0.08F} +
           horizon * Vec3{0.20F, 0.42F, 0.72F};
}

Vec3 floorColor(const Hit& hit, const Material& material) {
    const int x = static_cast<int>(std::floor(hit.position.x));
    const int z = static_cast<int>(std::floor(hit.position.z));
    const float checker = ((x + z) & 1) == 0 ? 1.0F : 0.55F;
    return material.albedo * checker;
}

Vec3 trace(const Ray& ray, const Scene& scene, int remainingBounces) {
    Hit hit;
    if (!intersectScene(ray, scene, kRayBias, kInfinity, hit)) {
        return skyColor(ray);
    }

    const Vec3 toLight = scene.lightPosition - hit.position;
    const float lightDistance = std::sqrt(dot(toLight, toLight));
    const Vec3 lightDirection = toLight / lightDistance;
    const Ray shadowRay{hit.position + hit.normal * kRayBias, lightDirection};
    Hit shadowHit;
    const bool inShadow = intersectScene(shadowRay, scene, kRayBias,
                                         lightDistance - kRayBias, shadowHit);
    const float diffuse = inShadow ? 0.0F : std::max(0.0F, dot(hit.normal, lightDirection));

    const Vec3 viewDirection = -ray.direction;
    const Vec3 halfVector = normalize(lightDirection + viewDirection);
    const float specular = inShadow ? 0.0F
                                    : std::pow(std::max(0.0F, dot(hit.normal, halfVector)), 64.0F);
    const Vec3 albedo = hit.isFloor ? floorColor(hit, hit.material) : hit.material.albedo;
    Vec3 color = albedo * (0.10F + 0.85F * diffuse) + Vec3{1.0F, 1.0F, 1.0F} * (0.30F * specular);

    if (remainingBounces > 0 && hit.material.reflectivity > 0.0F) {
        const Ray reflectedRay{
            hit.position + hit.normal * kRayBias,
            normalize(reflect(ray.direction, hit.normal)),
        };
        const Vec3 reflectedColor = trace(reflectedRay, scene, remainingBounces - 1);
        color = color * (1.0F - hit.material.reflectivity) +
                reflectedColor * hit.material.reflectivity;
    }

    return color;
}

std::uint32_t hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    value ^= value >> 16U;
    return value;
}

float random01(std::uint32_t seed) {
    return static_cast<float>(hash(seed) & 0x00FFFFFFU) / 16777216.0F;
}

Ray cameraRay(int pixelX, int pixelY, int sample, int width, int height) {
    const float jitterX = random01(static_cast<std::uint32_t>(pixelX) * 1973U +
                                   static_cast<std::uint32_t>(pixelY) * 9277U +
                                   static_cast<std::uint32_t>(sample) * 26699U);
    const float jitterY = random01(static_cast<std::uint32_t>(pixelX) * 3181U +
                                   static_cast<std::uint32_t>(pixelY) * 92821U +
                                   static_cast<std::uint32_t>(sample) * 1013U);
    const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    const float screenX = (2.0F * ((static_cast<float>(pixelX) + jitterX) /
                                    static_cast<float>(width)) - 1.0F) * aspectRatio;
    const float screenY = 1.0F - 2.0F * ((static_cast<float>(pixelY) + jitterY) /
                                          static_cast<float>(height));
    const Vec3 cameraPosition{0.0F, 0.25F, 5.8F};
    return {cameraPosition, normalize(Vec3{screenX, screenY - 0.10F, -1.65F})};
}

unsigned char toByte(float linearColor) {
    const float gammaCorrected = std::pow(clamp(linearColor, 0.0F, 1.0F), 1.0F / 2.2F);
    return static_cast<unsigned char>(255.0F * gammaCorrected + 0.5F);
}

void writePpm(const std::string& outputPath, const std::vector<Vec3>& pixels,
              int width, int height) {
    std::ofstream output(outputPath, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not open output file: " + outputPath);
    }

    output << "P6\n" << width << ' ' << height << "\n255\n";
    for (const Vec3& pixel : pixels) {
        const unsigned char rgb[] = {toByte(pixel.x), toByte(pixel.y), toByte(pixel.z)};
        output.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
    }
}

int parsePositiveInt(const char* value, const char* name) {
    try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) {
            throw std::invalid_argument("not positive");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string{name} + " must be a positive integer.");
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string outputPath = argc > 1 ? argv[1] : "ray_traced_circles.ppm";
        const int width = argc > 2 ? parsePositiveInt(argv[2], "Width") : 960;
        const int height = argc > 3 ? parsePositiveInt(argv[3], "Height") : 540;
        if (argc > 4) {
            std::cerr << "Usage: " << argv[0] << " [output.ppm] [width] [height]\n";
            return 1;
        }

        const Scene scene{
            {
                {{-1.55F, -0.60F, -1.8F}, 1.10F, {{0.95F, 0.18F, 0.16F}, 0.20F}},
                {{0.18F, -0.72F, -2.75F}, 1.00F, {{0.10F, 0.72F, 0.95F}, 0.30F}},
                {{1.82F, -0.94F, -3.85F}, 0.78F, {{0.98F, 0.75F, 0.10F}, 0.12F}},
            },
            {{0.0F, -1.72F, 0.0F}, {0.0F, 1.0F, 0.0F}, {{0.36F, 0.38F, 0.45F}, 0.28F}},
            {-3.0F, 5.5F, 3.5F},
        };

        std::vector<Vec3> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
        const unsigned int availableThreads = std::thread::hardware_concurrency();
        const unsigned int threadCount = std::max(1U, std::min(
            availableThreads == 0 ? 1U : availableThreads, static_cast<unsigned int>(height)));
        std::vector<std::thread> workers;
        workers.reserve(threadCount);

        for (unsigned int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
            workers.emplace_back([&, threadIndex] {
                const int firstRow = static_cast<int>(threadIndex * static_cast<unsigned int>(height) / threadCount);
                const int lastRow = static_cast<int>((threadIndex + 1U) * static_cast<unsigned int>(height) / threadCount);
                for (int y = firstRow; y < lastRow; ++y) {
                    for (int x = 0; x < width; ++x) {
                        Vec3 color{};
                        for (int sample = 0; sample < kSamplesPerPixel; ++sample) {
                            color += trace(cameraRay(x, y, sample, width, height), scene, kMaxBounces);
                        }
                        pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                               static_cast<std::size_t>(x)] = color / static_cast<float>(kSamplesPerPixel);
                    }
                }
            });
        }

        for (std::thread& worker : workers) {
            worker.join();
        }

        writePpm(outputPath, pixels, width, height);
        std::cout << "Rendered " << width << "x" << height << " using " << threadCount
                  << " CPU thread(s) to " << outputPath << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
