// 19-dipole.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cmath>
#include <vector>

double a = 1.1;  // nm
double dipole_strength = 0.08789;  // electron charge - nm
double eps_rel = 1.5;

double eps0 = 0.0552713;        // (electron charge) ^ 2 / (eV - nm)
double boltzmann = 8.617e-5;    // eV / K
double pi = 3.1415926535897932384626433832795028841971693993751058209749445923;

double k_un = 0.25 / (pi * eps0 * eps_rel);

struct {
    std::vector<double> x;
    std::vector<double> y;
} double_vector;

double_vector 

int main()
{
    std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
