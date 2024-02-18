#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/ray.h>
#include <filesystem/resolver.h>
#include <sh/spherical_harmonics.h>
#include <sh/default_image.h>
#include <Eigen/Core>
#include <fstream>
#include <random>
#include <stb_image.h>

NORI_NAMESPACE_BEGIN

namespace ProjEnv
{
    std::vector<std::unique_ptr<float[]>>
    LoadCubemapImages(const std::string &cubemapDir, int &width, int &height,
                      int &channel)
    {
        std::vector<std::string> cubemapNames{"negx.jpg", "posx.jpg", "posy.jpg",
                                              "negy.jpg", "posz.jpg", "negz.jpg"};
        std::vector<std::unique_ptr<float[]>> images(6);
        for (int i = 0; i < 6; i++)
        {
            std::string filename = cubemapDir + "/" + cubemapNames[i];
            int w, h, c;
            float *image = stbi_loadf(filename.c_str(), &w, &h, &c, 3);
            if (!image)
            {
                std::cout << "Failed to load image: " << filename << std::endl;
                exit(-1);
            }
            if (i == 0)
            {
                width = w;
                height = h;
                channel = c;
            }
            else if (w != width || h != height || c != channel)
            {
                std::cout << "Dismatch resolution for 6 images in cubemap" << std::endl;
                exit(-1);
            }
            images[i] = std::unique_ptr<float[]>(image);
            int index = (0 * 128 + 0) * channel;
            // std::cout << images[i][index + 0] << "\t" << images[i][index + 1] << "\t"
            //           << images[i][index + 2] << std::endl;
        }
        return images;
    }

    const Eigen::Vector3f cubemapFaceDirections[6][3] = { // mmc 6个平面的局部坐标系的、3个坐标轴方向
        {{0, 0, 1}, {0, -1, 0}, {-1, 0, 0}},  // negx // mmc negx和posx的x、y轴同向，z轴反向，说明一个是左手系一个是右手系
        {{0, 0, 1}, {0, -1, 0}, {1, 0, 0}},   // posx
        {{1, 0, 0}, {0, 0, -1}, {0, -1, 0}},  // negy
        {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}},    // posy
        {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, // negz
        {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},   // posz
    };

    float CalcPreArea(const float &x, const float &y)
    {
        return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0));
    }

    float CalcArea(const float &u_, const float &v_, const int &width,
                   const int &height)
    {
        // transform from [0..res - 1] to [- (1 - 1 / res) .. (1 - 1 / res)]
        // ( 0.5 is for texel center addressing)
        float u = (2.0 * (u_ + 0.5) / width) - 1.0;
        float v = (2.0 * (v_ + 0.5) / height) - 1.0;

        // shift from a demi texel, mean 1.0 / size  with u and v in [-1..1]
        float invResolutionW = 1.0 / width;
        float invResolutionH = 1.0 / height;

        // u and v are the -1..1 texture coordinate on the current face.
        // get projected area for this texel
        float x0 = u - invResolutionW;
        float y0 = v - invResolutionH;
        float x1 = u + invResolutionW;
        float y1 = v + invResolutionH;
        float angle = CalcPreArea(x0, y0) - CalcPreArea(x0, y1) -
                      CalcPreArea(x1, y0) + CalcPreArea(x1, y1);

        return angle;
    }

    // template <typename T> T ProjectSH() {}

    template <size_t SHOrder>
    std::vector<Eigen::Array3f> PrecomputeCubemapSH(const std::vector<std::unique_ptr<float[]>> &images,
                                                    const int &width, const int &height,
                                                    const int &channel)
    {
        std::vector<Eigen::Vector3f> cubemapDirs;
        cubemapDirs.reserve(6 * width * height);
        for (int i = 0; i < 6; i++)
        {
            Eigen::Vector3f faceDirX = cubemapFaceDirections[i][0];
            Eigen::Vector3f faceDirY = cubemapFaceDirections[i][1];
            Eigen::Vector3f faceDirZ = cubemapFaceDirections[i][2]; // mmc 朝向“外”的方向
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float u = 2 * ((x + 0.5) / width) - 1; // mmc -1到1
                    float v = 2 * ((y + 0.5) / height) - 1;
                    Eigen::Vector3f dir = (faceDirX * u + faceDirY * v + faceDirZ).normalized(); // mmc cube中心点到像素中心点的方向
                    cubemapDirs.push_back(dir);
                }
            }
        }
        constexpr int SHNum = (SHOrder + 1) * (SHOrder + 1); // mmc SHOrder是阶数
        std::vector<Eigen::Array3f> SHCoeffiecents(SHNum); // mmc 根据gpt，Eigen::Array3f相比Eigen::Vector3f更适用于逐元素操作，例如逐元素相加
        for (int i = 0; i < SHNum; i++)
            SHCoeffiecents[i] = Eigen::Array3f(0);
        float sumWeight = 0;
        for (int i = 0; i < 6; i++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // TODO: here you need to compute light sh of each face of cubemap of each pixel
                    // TODO: 此处你需要计算每个像素下cubemap某个面的球谐系数
                    Eigen::Vector3f dir = cubemapDirs[i * width * height + y * width + x];
                    int index = (y * width + x) * channel; // mmc 贴图的像素索引
                    Eigen::Array3f Le(images[i][index + 0], images[i][index + 1],
                                      images[i][index + 2]);
                    /* mmc 可以看到Le（入射光）就是读的贴图里的颜色，换句话说**环境贴图里存的是radiance，方向为从cubemap像素中心到cube中心**
                    * 我们知道对于mesh上的所有顶点，环境光sh系数用的都是相同的一套（不像light transport那样各顶点不同），也就是说认为**对于mesh上各点来说**，环境光照相同
                    * 更具体一点，假设mesh中心点为o，mesh上有两点a和b，cubemap某像素中心点为p，我们认为从p到o、从p到a、从p到b的radiance都相同
                    * 这个假设是合理的，因为环境光照本来就被认为是来自无限远处，可以理解成这种尺度下o、a、b是同一点，也可以理解成以o为中心的一个无限大cube、以a为中心的一个无限大cube、以b为中心的一个无限大cube是同一个cube
                    */

                    // Edit Start
                    auto delta_w = CalcArea(x, y, width, height);

                    for (int l = 0; l <= SHOrder; l++) {
                        for (int m = -l; m <= l; m++) {
                            auto basic_sh_proj = sh::EvalSH(l, m, Eigen::Vector3d(dir.x(), dir.y(), dir.z()).normalized()); // mmc 方向在基函数上的采样，返回一个double值
                            SHCoeffiecents[sh::GetIndex(l, m)] += Le * basic_sh_proj * delta_w; // mmc 对外面的三层循环（iyx）、也就是cubemap上的所有像素求和（黎曼积分）
                        }
                    }
                    // Edit End
                }
            }
        }
        return SHCoeffiecents; // mmc cubemap投影成SH系数
    }
}

class PRTIntegrator : public Integrator
{
public:
    static constexpr int SHOrder = 2;
    static constexpr int SHCoeffLength = (SHOrder + 1) * (SHOrder + 1);

    enum class Type
    {
        Unshadowed = 0,
        Shadowed = 1,
        Interreflection = 2
    };

    PRTIntegrator(const PropertyList &props)
    {
        /* No parameters this time */
        m_SampleCount = props.getInteger("PRTSampleCount", 100);
        m_CubemapPath = props.getString("cubemap");
        auto type = props.getString("type", "unshadowed");
        if (type == "unshadowed")
        {
            m_Type = Type::Unshadowed;
        }
        else if (type == "shadowed")
        {
            m_Type = Type::Shadowed;
        }
        else if (type == "interreflection")
        {
            m_Type = Type::Interreflection;
            m_Bounce = props.getInteger("bounce", 1);
        }
        else
        {
            throw NoriException("Unsupported type: %s.", type);
        }
    }

    std::unique_ptr<std::vector<double>> computeInterreflectionSH(Eigen::MatrixXf* directTSHCoeffs, const Point3f& pos, const Normal3f& normal, const Scene* scene, int bounces)
    {
        std::unique_ptr<std::vector<double>> coeffs(new std::vector<double>());
        coeffs->assign(SHCoeffLength, 0.0);

        if (bounces > m_Bounce)
            return coeffs;

        const int sample_side = static_cast<int>(floor(sqrt(m_SampleCount))); // mmc 以下参考自spherical_harmonics.cc，ProjectFunction
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> rng(0.0, 1.0);
        for (int t = 0; t < sample_side; t++) {
            for (int p = 0; p < sample_side; p++) {
                double alpha = (t + rng(gen)) / sample_side;
                double beta = (p + rng(gen)) / sample_side;
                double phi = 2.0 * M_PI * beta;
                double theta = acos(2.0 * alpha - 1.0);

                Eigen::Array3d d = sh::ToVector(phi, theta);
                const auto wi = Vector3f(d.x(), d.y(), d.z());
                double H = wi.normalized().dot(normal);
                Intersection its;
                if (H > 0.0 && scene->rayIntersect(Ray3f(pos, wi.normalized()), its)) // mmc 在上半球面击中
                {
                    MatrixXf normals = its.mesh->getVertexNormals(); // mmc 每列是一个顶点的法线
                    Point3f idx = its.tri_index; // mmc 就是一个float3，记录三角形三个顶点的顶点序号，应该就是击中的三角形
                    Point3f hitPos = its.p; // mmc 击中点空间位置
                    Vector3f bary = its.bary;

                    Normal3f hitNormal =
                        Normal3f(normals.col(idx.x()).normalized() * bary.x() + // mmc 看上去bary是顶点的（重心）权重
                            normals.col(idx.y()).normalized() * bary.y() +
                            normals.col(idx.z()).normalized() * bary.z())
                        .normalized(); // mmc 击中点法线的重心插值

                    auto nextBouncesCoeffs = computeInterreflectionSH(directTSHCoeffs, hitPos, hitNormal, scene, bounces + 1);

                    for (int i = 0; i < SHCoeffLength; i++)
                    {
                        auto interpolateSH = (directTSHCoeffs->col(idx.x()).coeffRef(i) * bary.x() +
                            directTSHCoeffs->col(idx.y()).coeffRef(i) * bary.y() +
                            directTSHCoeffs->col(idx.z()).coeffRef(i) * bary.z());

                        (*coeffs)[i] += (interpolateSH + (*nextBouncesCoeffs)[i]) * H; // mmc 推一下，是对的，注意computeInterreflectionSH（本函数）返回的结果是当前bounce及以后所有bounce的结果之和
                    }
                }
                // mmc else 没击中的认为(*coeffs)[i] += 0;
            }
        }

        for (unsigned int i = 0; i < coeffs->size(); i++) {
            (*coeffs)[i] /= sample_side * sample_side; // mmc 路径追踪，多条路径求平均
        }
        
        return coeffs;
    }

    virtual void preprocess(const Scene *scene) override
    {

        // Here only compute one mesh
        const auto mesh = scene->getMeshes()[0];
        // Projection environment
        auto cubePath = getFileResolver()->resolve(m_CubemapPath);
        auto lightPath = cubePath / "light.txt";
        auto transPath = cubePath / "transport.txt";
        std::ofstream lightFout(lightPath.str());
        std::ofstream fout(transPath.str());
        int width, height, channel;
        std::vector<std::unique_ptr<float[]>> images =
            ProjEnv::LoadCubemapImages(cubePath.str(), width, height, channel);
        auto envCoeffs = ProjEnv::PrecomputeCubemapSH<SHOrder>(images, width, height, channel);
        m_LightCoeffs.resize(3, SHCoeffLength); // mmc m_LightCoeffs全局，三通道；m_TransportSHCoeffs逐顶点，单通道
        for (int i = 0; i < envCoeffs.size(); i++)
        {
            lightFout << (envCoeffs)[i].x() << " " << (envCoeffs)[i].y() << " " << (envCoeffs)[i].z() << std::endl;
            m_LightCoeffs.col(i) = (envCoeffs)[i];
        }
        std::cout << "Computed light sh coeffs from: " << cubePath.str() << " to: " << lightPath.str() << std::endl;
        // Projection transport
        m_TransportSHCoeffs.resize(SHCoeffLength, mesh->getVertexCount()); // mmc 注意：为每个顶点计算light transfer的球面分布，也就是一套sh系数 // 但所有顶点共用一套环境光sh系数
        fout << mesh->getVertexCount() << std::endl;
        for (int i = 0; i < mesh->getVertexCount(); i++)
        {
            const Point3f &v = mesh->getVertexPositions().col(i);
            const Normal3f &n = mesh->getVertexNormals().col(i);
            auto shFunc = [&](double phi, double theta) -> double {
                Eigen::Array3d d = sh::ToVector(phi, theta); // mmc theta是与竖轴夹角，phi是平面上的角度
                const auto wi = Vector3f(d.x(), d.y(), d.z());
                // Edit Start
                double H = wi.normalized().dot(n.normalized());
                // Edit End
                if (m_Type == Type::Unshadowed)
                {
                    // TODO: here you need to calculate unshadowed transport term of a given direction
                    // TODO: 此处你需要计算给定方向下的unshadowed传输项球谐函数值
                    return H > 0.0 ? H : 0; // mmc 只要是上半球面，都认为可见
                }
                else
                {
                    // TODO: here you need to calculate shadowed transport term of a given direction
                    // TODO: 此处你需要计算给定方向下的shadowed传输项球谐函数值
                    if (H > 0.0 && !scene->rayIntersect(Ray3f(v, wi.normalized()))) {
                        return H;
                    }
                    return 0;
                } // mmc 这个函数求的是transfer项，也就是visibility和cosine两项的乘积，H就是cosine，而visibility只有0和1两个值，所以返回值就体现为H还是0二选一
            };
            auto shCoeff = sh::ProjectFunction(SHOrder, shFunc, m_SampleCount);
            for (int j = 0; j < shCoeff->size(); j++)
            {
                // Edit Start
                m_TransportSHCoeffs.col(i).coeffRef(j) = (*shCoeff)[j] / M_PI ; // mmc .coeffRef(j)看样子是Eigen::MatrixXf列内的索引方式（而非.row(j)） // 没明白这里为什么要除以pi，但是根据readme好像不除画面会过亮
                // Edit End
            }
        }
        if (m_Type == Type::Interreflection)
        {
            // TODO: leave for bonus

            for (int i = 0; i < mesh->getVertexCount(); i++)
            {
                const Point3f& v = mesh->getVertexPositions().col(i);
                const Normal3f& n = mesh->getVertexNormals().col(i).normalized();
                auto indirectCoeffs = computeInterreflectionSH(&m_TransportSHCoeffs, v, n, scene, 1);
                for (int j = 0; j < SHCoeffLength; j++)
                {
                    m_TransportSHCoeffs.col(i).coeffRef(j) += (*indirectCoeffs)[j]; // mmc 直接环境光 + 间接环境光
                }
                std::cout << "computing interreflection light sh coeffs, current vertex idx: " << i << " total vertex idx: " << mesh->getVertexCount() << std::endl;
            }
        }

        // Save in face format
        for (int f = 0; f < mesh->getTriangleCount(); f++)
        {
            const MatrixXu &F = mesh->getIndices();
            uint32_t idx0 = F(0, f), idx1 = F(1, f), idx2 = F(2, f);
            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx0).coeff(j) << " "; // mmc 一个顶点的transfer sh系数占一行
            }
            fout << std::endl;
            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx1).coeff(j) << " ";
            }
            fout << std::endl;
            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx2).coeff(j) << " ";
            }
            fout << std::endl;
        }
        std::cout << "Computed SH coeffs"
                  << " to: " << transPath.str() << std::endl;
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);

        const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1> sh0 = m_TransportSHCoeffs.col(its.tri_index.x()),
                                                                sh1 = m_TransportSHCoeffs.col(its.tri_index.y()),
                                                                sh2 = m_TransportSHCoeffs.col(its.tri_index.z());
        const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1> rL = m_LightCoeffs.row(0), gL = m_LightCoeffs.row(1), bL = m_LightCoeffs.row(2);

        Color3f c0 = Color3f(rL.dot(sh0), gL.dot(sh0), bL.dot(sh0)),
                c1 = Color3f(rL.dot(sh1), gL.dot(sh1), bL.dot(sh1)),
                c2 = Color3f(rL.dot(sh2), gL.dot(sh2), bL.dot(sh2));

        const Vector3f &bary = its.bary;
        Color3f c = bary.x() * c0 + bary.y() * c1 + bary.z() * c2;
        return c;
    }

    std::string toString() const
    {
        return "PRTIntegrator[]";
    }

private:
    Type m_Type;
    int m_Bounce = 1;
    int m_SampleCount = 100;
    std::string m_CubemapPath;
    Eigen::MatrixXf m_TransportSHCoeffs;
    Eigen::MatrixXf m_LightCoeffs;
};

NORI_REGISTER_CLASS(PRTIntegrator, "prt");
NORI_NAMESPACE_END