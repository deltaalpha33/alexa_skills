#include <iostream>
#include <fstream>
#include <string>
#include "boost/filesystem.hpp"   // includes all needed Boost.Filesystem declarations
#include <boost/foreach.hpp>
#include </opt/anaconda/include/python3.6m/Python.h>

namespace fs = boost::filesystem;
using namespace std;


int main () {
    cout << "hi";
    fs::path targetDir("/home/delta/mit-course/git/alexa_skills/sanitized_poetry_dataset/");

    fs::directory_iterator it(targetDir), eod;

    BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod))
    {
        if(fs::is_regular_file(p))
        {
            string document_dirs;
        }
    }

    return 0;
}

int
test(void)
{
  //Py_SetProgramName(argv[0]);  /* optional but recommended */
  Py_Initialize();
  PyRun_SimpleString("from time import time,ctime\n"
                     "print 'Today is',ctime(time())\n");
  Py_Finalize();
  return 0;
}
