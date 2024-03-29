//////////// To generate this program I refered https://raytracing.github.io/books/RayTracingInOneWeekend.html and other github  libraries
//////// I have taken the libraries and defined using many github programs I came across
//////// I modified and created most of the codes within


///**********************************************************************************************************


#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <array>
#include <functional>
#include <random>
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <fstream>
#include <future>

//=======================================================================================================================================
//---------------------------------------------------------------------------------------------------------------------------------------
using new_float = float;

//******************Modules for Vectors ***** I took reference form github to build similar operator definitions for vector**************

#define opr_vec_scal(opern, eq_opern) \
Type_Def& operator eq_opern (new_float value) { for(int i = 0; i < templ_dm; i++) v1.at(i) eq_opern value; return *this;} \
Type_Def operator opern (new_float value) const { return (Type_Def(*this) eq_opern value);}

#define opr_vec_vec(opern, eq_opern) \
Type_Def& operator eq_opern (const Type_Def& value) { for(int i = 0; i < templ_dm; i++) v1.at(i) eq_opern value.at(i); return *this;} \
Type_Def operator opern (const Type_Def& value) const { return (Type_Def(*this) eq_opern value);}

#define opr_scal_vec(Type_Def) \
inline Type_Def operator + (new_float left, const Type_Def& right) { return right + left;}  \
inline Type_Def operator - (new_float left, const Type_Def& right) { return -right + left;} \
inline Type_Def operator * (new_float left, const Type_Def& right) { return right * left;}

//---------------------------------------------------------------------------------------------------------------------------------------
template<uint8_t templ_dm>
class Vec
{
public:
    using Type_Def = Vec<templ_dm>;
    using Type_DefIn = const Vec<templ_dm>&;
    Vec() { for (auto& value : v1) value = 0; }
    Vec(const Type_Def& inside) :v1(inside.v1) {}
    Vec(std::initializer_list<new_float> init_lis)
    {
        if (init_lis.size() != templ_dm)
            throw std::runtime_error(std::string("Size didn't match!!!\n") + __FILE__);
        std::copy(init_lis.begin(), init_lis.end(), v1.begin());
    }

    const new_float& at(int i) const {
        return v1.at(i);
    }
    new_float& at(int i) {
        return v1.at(i);
    }

    opr_vec_scal(+, +=);
    opr_vec_scal(-, -=);
    opr_vec_scal(*, *=);
    opr_vec_scal(/ , /=);

    opr_vec_vec(+, +=);
    opr_vec_vec(-, -=);
    opr_vec_vec(*, *=);
    opr_vec_vec(/ , /=);

    Type_Def operator -() const {
        return Type_Def(*this) *= -1;
    }


    new_float Dot(Type_DefIn right) const
    {
        new_float Add(0);
        for (int i = 0; i < templ_dm; i++) Add += (this->at(i) * right.at(i));
        return Add;
    }

    new_float Norm() const {
        return sqrt(this->Dot(*this));
    }
    Type_Def Normalized() const {
        return (Type_Def(*this) /= this->Norm());
    }
    void Normalize() {
        *this /= this->Norm();
    }


protected:
    std::array<new_float, templ_dm> v1;
};

//---------------------------------------------------------------------------------------------------------------------------------------
#define Derived_opr_vec_scal(opern, eq_opern) \
Type_Def& operator eq_opern (new_float value) { BaseType::operator eq_opern (value); return *this;} \
Type_Def operator opern (new_float value) const { return BaseType::operator opern (value); }

#define Derived_opr_vec_vec(opern, eq_opern) \
Type_Def& operator eq_opern (const Type_Def& value) { BaseType::operator eq_opern (value); return *this;} \
Type_Def operator opern (const Type_Def& value) const { return BaseType::operator opern (value); }

//---------------------------------------------------------------------------------------------------------------------------------------
#define Derived_vec(this_type) \
    using BaseType = Vec; \
    using Type_Def = this_type; \
    this_type(std::initializer_list<new_float> init_lis): BaseType(init_lis){} \
    this_type(const BaseType& inside):BaseType(inside){} \
    this_type():BaseType(){} \
    Derived_opr_vec_scal(+, +=) \
    Derived_opr_vec_scal(-, -=) \
    Derived_opr_vec_scal(*, *=) \
    Derived_opr_vec_scal(/, /=) \
    Derived_opr_vec_vec(+, +=) \
    Derived_opr_vec_vec(-, -=) \
    Derived_opr_vec_vec(*, *=) \
    Type_Def operator -() const { return -(*this);}

//=======================================================================================================================================

//******************Modules for 3D Vectors *** I took reference form github to build similar operator definitions for vector*************
//=======================================================================================================================================
//---------------------------------------------------------------------------------------------------------------------------------------
class Vec3 : public Vec<3>
{
public:
    Derived_vec(Vec3);

    new_float& x() {
        return this->v1.at(0);
    }
    new_float& y() {
        return this->v1.at(1);
    }
    new_float& z() {
        return this->v1.at(2);
    }

    const new_float& x() const {
        return this->v1.at(0);
    }
    const new_float& y() const {
        return this->v1.at(1);
    }
    const new_float& z() const {
        return this->v1.at(2);
    }

    Vec3 Cross(const Vec3& right)
    {
        return Vec3{
            v1.at(1) * right.at(2) - v1.at(2) * right.at(1),
            -v1.at(0) * right.at(2) + v1.at(2) * right.at(0),
            v1.at(0) * right.at(1) - v1.at(1) * right.at(0) };
    }

    static Vec3 unit() { return Vec3{ 1,1,1 }; }

private:
};

opr_scal_vec(Vec3);

using V3_inside = const Vec3&;
using V3 = Vec3;



//============================================================Rays =======================================================================
// reference github multiple links

class Ray
{
public:
    Ray() {}
    Ray(V3_inside org, V3_inside direction, new_float minimum_t = 0, new_float maximum_t = 10000) :
        org1(org), direction1(direction), minimum_t_(minimum_t), maximum_t_(maximum_t) {}

    V3_inside org() const {
        return org1;
    }
    V3& org() {
        return org1;
    }

    V3_inside direction() const {
        return direction1;
    }
    V3& direction() {
        return direction1;
    }

    V3 operator() (new_float t) const {
        return org() + direction().Normalized() * t;
    }
    bool valid(new_float t) const {
        return t < maximum_t_&& t > minimum_t_;
    }
    new_float tMax() const {
        return maximum_t_;
    }
    new_float tMin() const {
        return minimum_t_;
    }

private:
    V3 org1;
    V3 direction1;
    new_float minimum_t_;
    new_float maximum_t_;
};

//=========================================================camera viewing ===========================================================
// reference github multiple links

class Camera_viewing
{
public:
    Camera_viewing(V3_inside Pos = V3(), V3_inside focus = V3{ 500, 500, 0 }, V3_inside c = V3{ 320, 240, 0 })
        :Pos1(Pos), f1(focus), c1(c) {}
    Ray Pix3Ray(int u, int v) const
    {
        auto dir = Vec3{ (u - c1.x()) / f1.x(),(v - c1.y()) / f1.y(), 1. };
        return Ray(Pos1, dir);
    }
private:
    Vec3 Pos1;
    Vec3 f1;
    Vec3 c1;
};

//======================================================Pixels ===========================================================================
// reference github multiple links

class Pix3 : public Vec<3>
{
public:
    Derived_vec(Pix3);

    new_float& red() {
        return this->v1.at(0);
    }
    new_float& green() {
        return this->v1.at(1);
    }
    new_float& blue() {
        return this->v1.at(2);
    }

    const new_float& red() const {
        return this->v1.at(0);
    }
    const new_float& green() const {
        return this->v1.at(1);
    }
    const new_float& blue() const {
        return this->v1.at(2);
    }

    int Red_place() const {
        return red() < 0 ? 0 : red() > 1 ? 255 : red() * 255.99;
    }
    int Green_place() const {
        return green() < 0 ? 0 : green() > 1 ? 255 : green() * 255.99;
    }
    int Blue_place() const {
        return blue() < 0 ? 0 : blue() > 1 ? 255 : blue() * 255.99;
    }

};

opr_scal_vec(Pix3);

//=======================================random number generate for pixels value======================================================
// reference github multiple links

namespace random_factory
{

    inline new_float Float_unit()
    {
        static std::uniform_real_distribution<double> distrib(0.0, 1.0);
        static std::mt19937 generator;
        static std::function<double()> genr_random =
            std::bind(distrib, generator);
        return genr_random();
    }

    inline Vec3 Sphere_rad_1()
    {
        Vec3 p;
        do
        {
            p = 2.0 * Vec3{ Float_unit(), Float_unit(), Float_unit() } - Vec3{ 1,1,1 };
        } while (p.Norm() > 1);
        return p;
    }

}

//==============================================================Rigid objects============================================================
// reference github multiple links
/// **************************************** I only defined single sphere here **********************************

class Rigid_Objects
{
public:
    struct save_hit
    {
        save_hit(new_float t_, const Vec3& p_, const Vec3& n_)
            :t(t_), p(p_), n(n_.Normalized()) {}
        new_float t; //hit t
        Vec3 p; //hit point
        Vec3 n; //Normal Vec
    };
    Rigid_Objects() {}

    enum Types
    {
        SPHERE
    };
    using Ptr_save_hit = std::shared_ptr<save_hit>;
    virtual Ptr_save_hit hit(const Ray& ray) const = 0;
    virtual std::string str() const { return ""; };

    static std::shared_ptr<Rigid_Objects> choose(Types type, const Vec3& center = V3(), const Vec3& size = V3{ 1,1,1 });
};

using Rigid_ObjectsPtr = std::shared_ptr<Rigid_Objects>;

class Sphere : public Rigid_Objects
{
public:
    Sphere(
        const Vec3& center = V3{ 0,0,0 },
        new_float radius = 3)
        :center_(center), radius_(radius) {}

    //checking the intersection of ray with sphere
   //*******************************************
   // quadratic polynomial equation
   //𝒗⋅𝒗𝑡^2+2𝒗⋅(𝒑−𝒄)𝑡+(𝒑−𝒄)⋅(𝒑−𝒄)−𝑟^2=0

    virtual Ptr_save_hit hit(const Ray& ray) const
    {
        auto oc = center_ - ray.org();
        auto unit_dir = ray.direction().Normalized();
        float dist = oc.Cross(unit_dir).Norm();
        if (dist >= radius_) return nullptr;
        auto t = oc.Dot(unit_dir) - sqrt(radius_ * radius_ - dist * dist);
        if (!ray.valid(t)) return nullptr;
        return std::make_shared<save_hit>(t, ray(t), ray(t) - center_);
    }
    virtual std::string str() const
    {
        return std::string("c: ") + std::to_string(center_.x()) + ", " + std::to_string(center_.y()) + ", " + std::to_string(center_.z());
    }
private:
    Vec3 center_;
    float radius_;
};

Rigid_ObjectsPtr Rigid_Objects::choose(Types type, const Vec3& center, const Vec3& size)
{
    if (type == Rigid_Objects::SPHERE)
        return std::make_shared<Sphere>(center, size.at(0));

    return std::make_shared<Sphere>();
}


//==============================================================Random Materials===========================================================
// I downloaded these along with the libraires from github multiple links


class Material
{
public:

    Material(const Vec3& albedo = V3()) :albedo_(albedo) {}

    const Pix3& attenuation() const { return albedo_; }
    virtual Ray scatter(
        const Ray& ray_inside, const Vec3& hit_p, const Vec3& hit_n) const = 0;

    enum Types
    {
        METAL,
        LAMBERTIAN,
        DIELECTRIC
    };
    static const new_float DEFUALT_FUZZ;
    static std::shared_ptr<Material> choose(Types type, const Pix3& albedo = V3{ .4, .4, .4 }, new_float fuzz = DEFUALT_FUZZ);

private:
    Pix3 albedo_;
};
using MaterialPtr = std::shared_ptr<Material>;

inline Vec3 reflect(V3_inside inside, V3_inside Norm)
{
    return inside - 2 * inside.Dot(Norm) * Norm;
}

inline std::shared_ptr<V3> refract(V3_inside dir_inside, V3_inside Normal, new_float ni_over_nt)
{

    new_float dt = dir_inside.Dot(Normal);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant <= 0) return nullptr;
    return std::make_shared<V3>(ni_over_nt * (dir_inside - Normal * dt) - Normal * sqrt(discriminant));
}

inline new_float schlick(new_float cosine, new_float ref_idx) {
    new_float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

const float Material::DEFUALT_FUZZ = -1;

class Lambertian :public Material
{
public:
    Lambertian(const Vec3& albedo, new_float fuzz) :Material(albedo), fuzz_(fuzz == DEFUALT_FUZZ ? 1. : fuzz) {}

    virtual Ray scatter(const Ray& ray_inside, const Vec3& hit_p, const Vec3& hit_n) const
    {
        Vec3 target = hit_p + hit_n + fuzz_ * random_factory::Sphere_rad_1();
        return Ray(hit_p, target - hit_p);
    }

private:
    new_float fuzz_;
};

class Metal :public Material
{
public:
    Metal(const Vec3& albedo, new_float fuzz) :Material(albedo), fuzz_(fuzz == DEFUALT_FUZZ ? 0 : fuzz) {}

    virtual Ray scatter(const Ray& ray_inside, const Vec3& hit_p, const Vec3& hit_n) const
    {
        Vec3 reflected = reflect(ray_inside.direction().Normalized(), hit_n);
        return Ray(hit_p, reflected + fuzz_ * random_factory::Sphere_rad_1());
    }
private:
    new_float fuzz_;
};

class Dielectric :public Material
{
public:
    Dielectric(new_float ri = 1.5) :Material(V3::unit()), ref_idx_(ri) {}

    virtual Ray scatter(const Ray& ray_inside, const Vec3& hit_p, const Vec3& hit_n) const
    {
        Vec3 outward_Normal;
        Vec3 reflected = reflect(ray_inside.direction(), hit_n);
        new_float ni_over_nt;

        new_float reflect_prob;
        new_float cosine;

        if (ray_inside.direction().Dot(hit_n) > 0) {
            outward_Normal = -hit_n;
            ni_over_nt = ref_idx_;
            cosine = ref_idx_ * ray_inside.direction().Normalized().Dot(hit_n);
        }
        else {
            outward_Normal = hit_n;
            ni_over_nt = 1.0 / ref_idx_;
            cosine = -ray_inside.direction().Normalized().Dot(hit_n);
        }
        auto refracted = refract(ray_inside.direction(), outward_Normal, ni_over_nt);

        reflect_prob = refracted ? schlick(cosine, ref_idx_) : 1.;

        return Ray(hit_p, (random_factory::Float_unit() < reflect_prob) ? reflected : *refracted);

    }
private:
    new_float ref_idx_;
};

MaterialPtr Material::choose(Types type, const Pix3& albedo, new_float fuzz)
{
    if (type == Material::METAL)
        return std::make_shared<Metal>(albedo, fuzz);
    else if (type == Material::LAMBERTIAN)
        return std::make_shared<Lambertian>(albedo, fuzz);
    else if (type == Material::DIELECTRIC)
        return std::make_shared<Dielectric>();

    return std::make_shared<Metal>(albedo, fuzz);
}

//==============================================================Hittable class ==========================================================
// I downloaded these along with the libraires from github multiple links
// To check if the ray of lights hits the sphere.



class Hittable
{
public:
    Hittable(Rigid_ObjectsPtr&& p_rigid, MaterialPtr&& p_material) :
        Rigid_Objects_(p_rigid),
        material_(p_material) {}

    Hittable(Hittable&& other) :
        Rigid_Objects_(other.Rigid_Objects_),
        material_(other.material_) {}

    Hittable(const Hittable& other) :
        Rigid_Objects_(other.Rigid_Objects_),
        material_(other.material_) {}

    // Hittable():
    //     Rigid_Objects_(Rigid_Objects::choose(Rigid_Objects::SPHERE)),
    //     material_(Material::choose(Material::METAL)){}

    const Rigid_Objects& Rigid_Objects() const { return *Rigid_Objects_; }
    const Material& material() const { return *material_; }

    struct save_hit {
        save_hit(Vec3 attenuation_, const Ray& scattered_)
            :attenuation(attenuation_), scattered(scattered_) {}
        Vec3 attenuation;
        Ray scattered;
        Vec3 n;
    };
    using Ptr_save_hit = std::shared_ptr<save_hit>;

protected:
    Rigid_ObjectsPtr Rigid_Objects_;
    MaterialPtr material_;

};

//=====================================================Hittablbe object manager===========================================================
// I downloaded these along with the libraires from multiple github links


class HitManager
{
public:

    Hittable::Ptr_save_hit hit(const Ray& ray) const
    {
        Rigid_Objects::Ptr_save_hit closest_hit = nullptr;
        const Hittable* closest_obj = nullptr;
        for (const auto& obj : hittables1)
        {
            auto record = obj.Rigid_Objects().hit(ray);
            if (record == nullptr) continue;
            if ((nullptr == closest_hit && nullptr == closest_obj)
                || record->t < closest_hit->t)
            {
                closest_hit = record;
                closest_obj = &obj;
            }

        }
        if (nullptr == closest_obj)
        {
            // std::cout << "obj: " << hittables1.front().Rigid_Objects().str() << std::endl;
            return nullptr;
        }
        auto ret = std::make_shared<Hittable::save_hit>(
            closest_obj->material().attenuation(),
            closest_obj->material().scatter(ray, closest_hit->p, closest_hit->n));
        ret->n = closest_hit->n;
        return ret;
    }

    void hittables1push(Rigid_ObjectsPtr&& p_rigid, MaterialPtr&& p_material)
    {
        hittables1.push_back(Hittable(std::forward<Rigid_ObjectsPtr>(p_rigid), std::forward<MaterialPtr>(p_material)));
    }

private:
    // std::vector<HittablePtr> hittables1;
    std::vector<Hittable> hittables1;
};

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//_____________________writting and reading from files_

template<size_t CH_NUM>
std::vector<std::array<float, CH_NUM>>
LoadDataFromFile(const std::string& filename)
{
    std::vector<std::array<float, CH_NUM>> returnvalue;
    std::ifstream fin(filename);

    if (fin.is_open()) {
        while (!fin.eof()) {
            std::array<float, CH_NUM> Read_line;

            for (size_t i = 0; i < CH_NUM; i++)
                fin >> Read_line[i];
            returnvalue.push_back(Read_line);
        }
    }
    return returnvalue;
}


using Pix3Coordinates = std::vector<std::array<int, 2>>;

//writing to targa file
void writeimage_TGA(const std::string& filename, int width, int height, const std::vector<Pix3>& pix_val)
{
    uint8_t FileChanVal = 3; //file channels value
    if (pix_val.size() != width * height)
    {
        std::cout << "size mismatch! width:" << width << " and " << " height: " << height
            << ", width*height: " << width * height << ", pix_val.size(): " << pix_val.size() << std::endl;
    }
    FILE* file_ptr = NULL;
    file_ptr = fopen(filename.c_str(), "wb");
    if (file_ptr == NULL) return;

    uint8_t header[18] = { 0,0,2,0,0,0,0,0,0,0,0,0, (uint8_t)(width % 256), (uint8_t)(width / 256), (uint8_t)(height % 256), (uint8_t)(height / 256), (uint8_t)(FileChanVal * 8), 0x0 };
    fwrite(&header, 18, 1, file_ptr);

    for (const auto& px : pix_val)
    {
        putc((int)(px.Blue_place()), file_ptr);
        putc((int)(px.Green_place()), file_ptr);
        putc((int)(px.Red_place()), file_ptr);
    }

    fclose(file_ptr);
}

inline Pix3Coordinates Coordinate_seq(int width, int height)
{
    Pix3Coordinates returnvalue;
    for (int j = height - 1; j >= 0; j--)
    {
        for (int i = 0; i < width; i++)
        {
            returnvalue.push_back(std::array<int, 2>{i, j});
        }
    }
    return returnvalue;
}
//**********************************dist betn ray and point ***************************************************************************

float Ray_Point_dist(const Ray& ray, const Vec3& pt)
{
    float org_pt_dist = (ray.org() - pt).Norm();
    Vec3 line = (pt - ray.org());
    float angle = acos(line.Normalized().Dot(ray.direction().Normalized()));
    float distance = org_pt_dist * sin(angle);
    return distance;
}

Pix3 lightSource(const Ray& ray)
{
    std::vector<Vec3> points;
    auto data_lights = LoadDataFromFile<6>("lights.txt");
    for (const auto& light : data_lights)
    {
        points.push_back(Vec3{ light[0],light[1],light[2] });
    }
    // points.resize(20);

    for (auto& pt : points)
    {
        if (Ray_Point_dist(ray, pt) < 20) return Pix3{ 1.,1.,1. } *0.6;
    }
    return Pix3{ 0,0,0. };
}

Pix3 lightContribution(const HitManager& manager, const Vec3& hit_p, const Pix3& obj_color, const Vec3& surface_Norm)
{
    std::vector<Vec3> points;
    auto data_lights = LoadDataFromFile<6>("lights.txt");
    for (const auto& light : data_lights)
    {
        points.push_back(Vec3{ light[0],light[1],light[2] });
    }

    Pix3 ret_color;
    float contri_coeff = 0.f;
    for (const auto& light_pt : points)
    {
        Vec3 hit_point_to_light = light_pt - hit_p;
        Ray test_ray(hit_p, hit_point_to_light.Normalized());
        auto hit_record = manager.hit(test_ray);
        if (hit_record != nullptr) continue; // not visible to light

        float light_hitpt_dist = hit_point_to_light.Norm();
        float local_coeff = surface_Norm.Dot(hit_point_to_light.Normalized()) / (light_hitpt_dist * light_hitpt_dist);
        if (local_coeff <= 0) continue;

        contri_coeff += local_coeff * 8000;

    }

    ret_color = obj_color * (contri_coeff / points.size());
    return ret_color;
}

Pix3 color_assign(const HitManager& manager, const Ray& R1, int depth)
{
    auto p_record = manager.hit(R1);
    if (p_record)
    {
        auto obj_color = p_record->attenuation;
        return lightContribution(manager, p_record->scattered.org(), obj_color, p_record->n);
        manager.hit(R1);
    }
    return Pix3{ 0.6, 0.6, 0.6 };

}


std::vector<Pix3> threadFunc(
    const Camera_viewing& cam1,
    const HitManager& manager,
    int Th_sample_numr,
    Pix3Coordinates::const_iterator begin,
    Pix3Coordinates::const_iterator end)
{
    std::vector<Pix3> returnval;
    for (auto it = begin; it != end; it++)
    {
        auto& uv = *it;
        Pix3 col;
        auto r = cam1.Pix3Ray(uv[0], uv[1]);
        for (int s = 0; s < Th_sample_numr; s++)
        {

            col += color_assign(manager, r, 0);
        }
        col /= float(Th_sample_numr);
        for (int i = 0; i < 3; i++) col.at(i) = sqrt(col.at(i));
        returnval.push_back(col);
    }
    return returnval;
}




//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++MAIN++++++++++++++++++++++++++++++++++++++++++++++
//MAin---------------------------------------------Main---------------------------
int main(int argc, char const* argv[])
{

    std::cout << "*****************************************************************";
    int N = 1024;
    int nx = N;  
    int ny = N;

    int Th_sample_numr = 1;

    Camera_viewing cam1(Vec3(), Vec3{ 500, 500, 0 }, Vec3{ (new_float)nx / 2, (new_float)ny / 2, 0 });
    std::vector<Pix3> img_for_pixel;

    HitManager manager;

    auto sphere_infos = LoadDataFromFile<7>("spheres.txt");
    for (const auto& s : sphere_infos)
        manager.hittables1push(Rigid_Objects::choose(Rigid_Objects::SPHERE, V3{ s[0], s[1], s[2] }, V3::unit() * s[3]), Material::choose(Material::LAMBERTIAN, Pix3{ s[4], s[5], s[6] }, 1.));

    // for multiprocessing  
    bool multi_processing = 1;


    auto Coordinate = Coordinate_seq(nx, ny);


    if (multi_processing)
    {
        int thread_numer = std::thread::hardware_concurrency() * 0.9;

        // similar to Threadpool pool(thread_numer);
        std::vector< std::future<std::vector<Pix3>> > pool_results;

        int StepLength = 10000;

        // we resize as pool_results.resize(Coordinate.size() / StepLength + 1);

        int True_size = 0;
        for (int i = 0; i < Coordinate.size(); i += StepLength)
        {
            pool_results.push_back(
                std::async(&threadFunc, cam1, manager, Th_sample_numr, Coordinate.begin() + i,
                    i + StepLength >= Coordinate.size() ? Coordinate.end() : Coordinate.begin() + i + StepLength)
            );
        }
        // actual size ... pool_results.resize(True_size);

        for (auto& vec : pool_results)
        {
            for (auto& px : vec.get())
                img_for_pixel.push_back(px);

            if (img_for_pixel.size() % 1000 == 0)
                std::cout << "Rendering done: " << 100. * img_for_pixel.size() / Coordinate.size() << "%" << std::endl;
        }
    }
    else
    {
        img_for_pixel = threadFunc(cam1, manager, Th_sample_numr, Coordinate.begin(), Coordinate.end());
    }

    writeimage_TGA("output.tga", nx, ny, img_for_pixel);
    return 0;
}
