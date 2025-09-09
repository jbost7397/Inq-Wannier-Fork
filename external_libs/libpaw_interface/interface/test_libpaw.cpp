#include <catch2/catch_all.hpp>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

std::vector<double> readDataFromFile(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    double value;

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

double calculateSumOfAbsoluteDifferences(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    double sum = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        sum += std::abs(vec1[i] - vec2[i]);
    }
    return sum / vec1.size();
}

const double threshold = 1e-9;

const std::string outputDir = "./";
const std::string referenceDir = "./reference_data/";

const std::vector<std::string> fileNames = {
    "dij.dat",
    "ncoret.dat",
    "nhat.dat",
    "vloc.dat"
};

TEST_CASE("FortranLibraryTests", "[fortran]") {
    for (const std::string& fileName : fileNames) {
        SECTION("Testing " + fileName) {
            std::vector<double> outputData = readDataFromFile(outputDir + fileName);
            std::vector<double> referenceData = readDataFromFile(referenceDir + fileName);
            double sumOfDifferences = calculateSumOfAbsoluteDifferences(outputData, referenceData);
            REQUIRE(sumOfDifferences < threshold);
        }
    }
}