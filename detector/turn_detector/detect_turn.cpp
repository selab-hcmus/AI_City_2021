#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>

namespace Geometry {
  const double eps = 0.04;

  template<typename T>
  struct Point {
    T x;
    T y;

    inline Point operator+(const Point& rhs) const {
      return {x + rhs.x, y + rhs.y};
    }
  };

  template<typename T>
  inline std::istream& operator>>(std::istream& is, Point<T>& p) {
    return is >> p.x >> p.y;
  }  

  template<typename T>
  inline std::ostream& operator<<(std::ostream& os, const Point<T>& p) {
    return os << p.x << ' ' << p.y;
  }

  template<typename T>
  inline int sign(const T& x) {
    return x < -eps ? -1 : (x > eps ? 1 : 0);
  }

  template<typename T>
  inline T sqr(const T& x) {
    return x * x;
  }

  template<typename T>
  inline T cross(const Point<T>& p, const Point<T>& q) {
    return p.x * q.y - p.y * q.x;
  }

  template<typename T>
  inline T dist2(const Point<T>& p, const Point<T>& q) {
    return sqr(p.x - q.x) + sqr(p.y - q.y);
  }

  template<typename T>
  inline T area2(const Point<T>& a, const Point<T>& b, const Point<T>& c) {
    return cross(a, b) + cross(b, c) + cross(c, a);
  }
}

using Point = Geometry::Point<long long>;

#define TRACE 0
#define TEST_FOLDER "test_track_boxes_ver2"
#define RESULTS_FOLDER "results"

class Polyline {
  std::vector<Point> points;
  std::string camera_id;

public:
  Polyline(const std::string& filename) {
    std::ifstream in(filename);
    int n;
    in >> n;
    points.resize(n);
    in >> camera_id;
    for (auto& p : points) {
      Point a, b;
      in >> a >> b;
      p = a + b;
      p.y = -p.y;
    }
  }

  double get_direction(std::string filename) const {
    std::ofstream out(RESULTS_FOLDER"/" + filename);
    long long distance2 = dist2(points.front(), points.back());
    long long algebra_area = 0;
    for (int i = 2; i < points.size(); ++i) {
      algebra_area += area2(points.front(), points[i - 1], points[i]);
    }
    double value = algebra_area / 2.0 / distance2;
#if TRACE
    for (const auto& p : points) {
      out << p.x << ' ' << p.y << '\n';
    }
    out << camera_id << '\n';
    out << value << '\n';
#endif
    out << Geometry::sign(value) << std::endl;
    return value;
  }
};

std::vector<std::pair<std::string, double>> scores;
constexpr double eps = 0.036;

void print_turns() {
  std::set<int> turns;
  for (auto& it : scores) {
    if (fabs(it.second) > eps) {
      auto i = it.first.end();
      it.first.erase(i - 4, i);
      turns.insert(std::stoi(it.first));
    }    
  }
  for (int x : turns) {
    std::cout << x << '\n';
  }
}

int main() {  
  system("dir /B " TEST_FOLDER" > list.txt");
  system("mkdir " RESULTS_FOLDER);

  std::ifstream in("list.txt");
  std::string filename;
  while (in >> filename) {
    double score = Polyline(TEST_FOLDER"/" + filename).get_direction(filename);
    scores.push_back({filename, score});
  }
  print_turns();
  return 0;
}