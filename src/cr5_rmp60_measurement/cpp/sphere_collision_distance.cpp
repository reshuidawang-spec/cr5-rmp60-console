// Minimal FCL/Assimp bridge. Distances and vertices use millimetres.
#include <fcl/fcl.h>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shapes.h>
#include <memory>
#include <vector>
#include <cmath>
#include <exception>
#include <limits>
using Geometry=std::shared_ptr<fcl::CollisionGeometryd>;
extern "C" {
void* cr5_shape(int type,const double* dimensions) {
  try {
    Geometry shape;
    if(type==1) shape=std::make_shared<fcl::Sphered>(dimensions[0]);
    else if(type==2) shape=std::make_shared<fcl::Cylinderd>(dimensions[0],dimensions[1]);
    else if(type==3) shape=std::make_shared<fcl::Boxd>(dimensions[0],dimensions[1],dimensions[2]);
    else if(type==4) shape=std::make_shared<fcl::Capsuled>(dimensions[0],dimensions[1]);
    else return nullptr;
    return new Geometry(shape);
  } catch (...) {return nullptr;}
}
void* cr5_convex(const double* vertices,int count,const int* faces,int face_count,int face_size) {
  try {
    auto points=std::make_shared<std::vector<fcl::Vector3d>>();
    for(int i=0;i<count;++i) points->emplace_back(vertices[i*3],vertices[i*3+1],vertices[i*3+2]);
    auto polygons=std::make_shared<std::vector<int>>(faces,faces+face_size);
    return new Geometry(std::make_shared<fcl::Convexd>(points,face_count,polygons,true));
  } catch (...) {return nullptr;}
}
// Two-pass API; Uses the exact same geometric_shapes mesh loader as MoveIt.
int cr5_mesh_vertices(const char* filename,double* output,int capacity) {
  try {
    std::unique_ptr<shapes::Mesh> mesh(shapes::createMeshFromResource(std::string("file://")+filename));
    if(!mesh) return -1;
    int count=mesh->vertex_count;
    if(!output) return count;
    if(capacity<count) return -1;
    std::copy(mesh->vertices,mesh->vertices+count*3,output);
    return count;
  } catch (...) {return -1;}
}
void cr5_free(void* handle) {delete static_cast<Geometry*>(handle);}
double cr5_distance(void* a,const double* ta,void* b,const double* tb) {
  try {
    fcl::Transform3d t1=fcl::Transform3d::Identity(),t2=t1;
    for(int i=0;i<3;++i) {
      for(int j=0;j<3;++j) {t1.linear()(i,j)=ta[i*4+j];t2.linear()(i,j)=tb[i*4+j];}
      t1.translation()[i]=ta[i*4+3];t2.translation()[i]=tb[i*4+3];
    }
    fcl::CollisionObjectd first(*static_cast<Geometry*>(a),t1),second(*static_cast<Geometry*>(b),t2);
    fcl::CollisionRequestd creq; fcl::CollisionResultd cres;
    fcl::collide(&first,&second,creq,cres);
    if(cres.isCollision()) return -1.;
    fcl::DistanceRequestd request;request.gjk_solver_type=fcl::GST_LIBCCD;
    fcl::DistanceResultd result;
    double d=fcl::distance(&first,&second,request,result);
    return std::isfinite(d) && d<std::numeric_limits<double>::max()/2 ? d:NAN;
  } catch (...) {return NAN;}
}
}
