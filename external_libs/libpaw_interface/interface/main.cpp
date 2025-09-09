#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <vector>

extern "C"
{
    void fortran_main_(double&,double&,double*,double*,double*,double&,int*,int*,
    //                 ecut    ecutpaw gmet    rprimd  gprimd  ucvol   ngfft ngfftdg
        int&,int&,int*,double*,char*);
    //  natom ntypat typat xred filename_list
}

// some helper functions which are only used here for testing
// as all the data will be passed from the main DFT software
// in actual application

const char* ws = " \t\n\r\f\v";

// trim from end of string (right)
inline std::string& rtrim(std::string& s, const char* t = ws)
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim(std::string& s, const char* t = ws)
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim(std::string& s, const char* t = ws)
{
    return ltrim(rtrim(s, t), t);
}

inline void scan_input_double_scalar(std::ifstream & ifs_in, std::string key_in, double & val)
{
    std::string key;
    
    ifs_in >> key >> val;
    assert(trim(key_in) == trim(key));
}

inline void scan_input_double(std::ifstream & ifs_in, std::string key_in, std::vector<double> & val, const int size)
{
    val.resize(size);

    std::string key;
    ifs_in >> key;
    assert(trim(key_in) == trim(key));

    for(int i = 0; i < size; i ++)
    {
        ifs_in >> val[i];
    }
}

inline void scan_input_int_scalar(std::ifstream & ifs_in, std::string key_in, int & val)
{
    std::string key;
    
    ifs_in >> key >> val;
    assert(trim(key_in) == trim(key));
}

inline void scan_input_int(std::ifstream & ifs_in, std::string key_in, std::vector<int> & val, const int size)
{
    val.resize(size);

    std::string key;
    ifs_in >> key;
    assert(trim(key_in) == trim(key));

    for(int i = 0; i < size; i ++)
    {
        ifs_in >> val[i];
    }
}

int main()
{
    std::ifstream input("input");

    double ecut, ecutpaw;
    std::vector<double> gmet(9), rprimd(9), gprimd(9);
    double ucvol;
    std::vector<int> ngfft(3), ngfftdg(3);
    int natom,ntypat;
    std::vector<int> typat;
    std::vector<double> xred;

    char* filename_list;

    scan_input_double_scalar(input,"ecut",ecut);
    scan_input_double_scalar(input,"ecutpaw",ecutpaw);
    scan_input_double(input,"gmet",gmet,9);
    scan_input_double(input,"rprimd",rprimd,9);
    scan_input_double(input,"gprimd",gprimd,9);
    scan_input_double_scalar(input,"ucvol",ucvol);
    scan_input_int(input,"ngfft",ngfft,3);
    scan_input_int(input,"ngfftdg",ngfftdg,3);
    scan_input_int_scalar(input,"natom",natom);
    scan_input_int_scalar(input,"ntypat",ntypat);
    
    scan_input_int(input,"typat",typat,natom);
    scan_input_double(input,"xred",xred,3*natom);

    filename_list = new char[ntypat*264];

    std::ifstream pawfile("pawfiles");
    for(int i = 0; i < ntypat; i++)
    {
        pawfile.getline(&filename_list[264*i],264);
    }

    fortran_main_(ecut,ecutpaw,gmet.data(),rprimd.data(),gprimd.data(),
        ucvol,ngfft.data(),ngfftdg.data(),natom,ntypat,typat.data(),xred.data(),filename_list);
}