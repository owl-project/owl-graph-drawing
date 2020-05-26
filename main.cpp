// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

#include <boost/filesystem.hpp>

#include <GL/glew.h>

#include <visionaray/gl/debug_callback.h>
#include <visionaray/gl/handle.h>
#include <visionaray/gl/program.h>
#include <visionaray/gl/shader.h>
#include <visionaray/math/io.h>
#include <visionaray/pinhole_camera.h>

#include <common/input/key_event.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/timer.h>
#include <common/viewer_glut.h>
#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>
#include <Support/StringSplit.h>

#define VERBOSE 1

#include "gd.h"
#include "graph.h"

using namespace support;

using namespace visionaray;

//-------------------------------------------------------------------------------------------------
// Misc.helpers
//

std::istream& operator>>(std::istream& in, pinhole_camera& cam)
{
    vec3 eye;
    vec3 center;
    vec3 up;

    in >> eye >> std::ws >> center >> std::ws >> up >> std::ws;
    cam.look_at(eye, center, up);

    return in;
}

std::ostream& operator<<(std::ostream& out, pinhole_camera const& cam)
{
    out << cam.eye() << '\n';
    out << cam.center() << '\n';
    out << cam.up() << '\n';
    return out;
}


//-------------------------------------------------------------------------------------------------
// Cmdline parsing
//

enum eInputSource {
    ARTIFICIAL,
    TREE,
    FILEINPUT
};

struct ArtificialOptions {
    int Clusters = 80;
    int NodesPerCluster = 50;
    int EdgesPerCluster = 100;
    bool Connected = true;
};

struct TreeOptions {
    int depth=1;
    int degree=2;
};

struct Cmd {
    std::set<std::string> inputFiles;
    std::string sOutputFile = "";
    int maxIterations = -1;
    int loadRepetitions = 1;
    bool active=false;
    bool benchmarkMode=false;

    eInputSource inputSource = eInputSource::ARTIFICIAL;

    LayouterMode layouterMode=LayouterMode::RTX;

    unsigned rebuildRTXAccelAfter=1;

    ArtificialOptions artificialOptions;
    TreeOptions treeOptions;
    std::string initial_camera;
};


struct renderer : visionaray::viewer_glut
{
    struct GraphPipeline
    {
        gl::buffer  vertex_buffer;
        gl::buffer  index_buffer;
        gl::program prog;
        gl::shader  vert;
        gl::shader  frag;
        GLuint      view_loc;
        GLuint      proj_loc;
        GLuint      vertex_loc;
    };

    GraphPipeline graphPipeline;

    bool buildGraphPipeline()
    {
        // Setup shaders
        graphPipeline.vert.reset(glCreateShader(GL_VERTEX_SHADER));
        graphPipeline.vert.set_source(R"(
            attribute vec2 vertex;

            uniform mat4 view;
            uniform mat4 proj;


            void main(void)
            {
                gl_Position = proj * view * vec4(vertex, 0.0, 1.0);
            }
            )");
        graphPipeline.vert.compile();
        if (!graphPipeline.vert.check_compiled())
        {
            return false;
        }

        graphPipeline.frag.reset(glCreateShader(GL_FRAGMENT_SHADER));
        graphPipeline.frag.set_source(R"(
            void main(void)
            {
                gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
            )");
        graphPipeline.frag.compile();
        if (!graphPipeline.frag.check_compiled())
        {
            return false;
        }

        graphPipeline.prog.reset(glCreateProgram());
        graphPipeline.prog.attach_shader(graphPipeline.vert);
        graphPipeline.prog.attach_shader(graphPipeline.frag);

        graphPipeline.prog.link();
        if (!graphPipeline.prog.check_linked())
        {
            return false;
        }

        graphPipeline.vertex_loc = glGetAttribLocation(graphPipeline.prog.get(), "vertex");
        graphPipeline.view_loc   = glGetUniformLocation(graphPipeline.prog.get(), "view");
        graphPipeline.proj_loc   = glGetUniformLocation(graphPipeline.prog.get(), "proj");


        // Setup vbo
        graphPipeline.vertex_buffer.reset(gl::create_buffer());
        graphPipeline.index_buffer.reset(gl::create_buffer());

        return true;
    }

    void updateGraphPipeline()
    {
        // Store OpenGL state
        GLint array_buffer_binding = 0;
        GLint element_array_buffer_binding = 0;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer_binding);
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &element_array_buffer_binding);

        glBindBuffer(GL_ARRAY_BUFFER, graphPipeline.vertex_buffer.get());
        glBufferData(GL_ARRAY_BUFFER,
                     graphBack.nodes.size() * sizeof(vec2),
                     graphBack.nodes.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphPipeline.index_buffer.get());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     graphBack.edges.size() * sizeof(vec2ui),
                     graphBack.edges.data(),
                     GL_STATIC_DRAW);

        // Restore OpenGL state
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_array_buffer_binding);
        glBindBuffer(GL_ARRAY_BUFFER, array_buffer_binding);
    }

    renderer(bool active=false, bool benchmarkMode=false)
        : viewer_glut(1024, 1024, "Spring Embedder")
        , active(active)
    {
        genCommandLineOptions();
    }

    void genCommandLineOptions();

    void on_display()
    {
        if (!glInitialized)
        {
            glewInit();

            glClearColor(1,1,1,1);
            glEnable(GL_DEPTH_TEST);

            glInitialized = buildGraphPipeline();

            debugcb.activate();
        }

        if (imageUpdated)
        {
            std::unique_lock<std::mutex> l(mtx);

            graphBack = graph;

            updateGraphPipeline();

            imageUpdated = false;
        }

        // Store OpenGL state
        GLint array_buffer_binding = 0;
        GLint element_array_buffer_binding = 0;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer_binding);
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &element_array_buffer_binding);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw buffers

        graphPipeline.prog.enable();

        glUniformMatrix4fv(graphPipeline.view_loc, 1, GL_FALSE, cam.get_view_matrix().data());
        glUniformMatrix4fv(graphPipeline.proj_loc, 1, GL_FALSE, cam.get_proj_matrix().data());

        glBindBuffer(GL_ARRAY_BUFFER, graphPipeline.vertex_buffer.get());
        glEnableVertexAttribArray(graphPipeline.vertex_loc);
        glVertexAttribPointer(graphPipeline.vertex_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphPipeline.index_buffer.get());

        glDrawElements(GL_LINES, graphBack.edges.size()*2, GL_UNSIGNED_INT, 0);

        glDisableVertexAttribArray(graphPipeline.vertex_loc);

        graphPipeline.prog.disable();

        // Restore OpenGL state
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_array_buffer_binding);
        glBindBuffer(GL_ARRAY_BUFFER, array_buffer_binding);
    }

    void on_key_press(visionaray::key_event const& event)
    {
        if (cmd.benchmarkMode)
        {
            visionaray::viewer_glut::on_key_press(event);
            return;
        }

        static const std::string camera_file_base = "visionaray-camera";
        static const std::string camera_file_suffix = ".txt";
        
        switch (event.key())
        {
        case visionaray::keyboard::Space:
            {
                std::unique_lock<std::mutex> l(mtx);
                active = !active;
            }
            break;

        case 'p':
            {
                graphBack.printStatistics(true);
            }
            break;

        case 's':
            {
                std::cout << "Save tulip dump\n";
                //if (!cmd.sOutputFile.empty())
                //    graphBack.saveTLP(cmd.sOutputFile);
                //else
                    graphBack.saveTLP("dump.tlp");
            }
            break;

        case 'u':
            {
                int inc = 0;
                std::string inc_str = "";

                std::string filename = camera_file_base + inc_str + camera_file_suffix;

                while (boost::filesystem::exists(filename))
                {
                    ++inc;
                    inc_str = std::to_string(inc);

                    while (inc_str.length() < 4)
                    {
                        inc_str = std::string("0") + inc_str;
                    }

                    inc_str = std::string("-") + inc_str;

                    filename = camera_file_base + inc_str + camera_file_suffix;
                }

                std::ofstream file(filename);
                if (file.good())
                {
                    std::cout << "Storing camera to file: " << filename << '\n';
                    file << cam;
                }
            }
            break;

        case 'v':
            {
                std::string filename = camera_file_base + camera_file_suffix;

                load_camera(filename);
            }
            break;
        }

        visionaray::viewer_glut::on_key_press(event);
    }

    void on_close()
    {
        doQuit = true;

        visionaray::viewer_glut::on_close();
    }

    void load_camera(std::string filename)
    {
        std::ifstream file(filename);
        if (file.good())
        {
            file >> cam;
            std::cout << "Load camera from file: " << filename << '\n';
        }
    }


    Graph graph;
    Graph graphBack;

    bool glInitialized = false;

    gl::debug_callback debugcb;

    bool imageUpdated = false;

    bool active = false;

    bool doQuit = false;

    std::mutex mtx;

    pinhole_camera cam;

    timer total_time;

    Cmd cmd;
};



void renderer::genCommandLineOptions()
{
    add_cmdline_option(cl::makeOption<std::string&>(
        cl::Parser<>(),
        "camera",
        cl::Desc("Text file with camera parameters"),
        cl::ArgRequired,
        cl::init(cmd.initial_camera)
        ) );

    add_cmdline_option(cl::makeOption<int&>(
        cl::Parser<>(), "n",
        cl::ArgName("int"),
        cl::ArgOptional,
        cl::init(cmd.maxIterations),
        cl::Desc("Maximum number of iterations")
        ));
   
    add_cmdline_option(cl::makeOption<int&>(
        cl::Parser<>(), "r",
        cl::ArgName("int"),
        cl::ArgOptional,
        cl::init(cmd.loadRepetitions),
        cl::Desc("Number of repetitions when loading data")
        ));

    add_cmdline_option(cl::makeOption<bool&>(
        cl::Parser<>(), "bench",
        cl::ArgName("bool"),
        cl::ArgOptional,
        cl::init(cmd.benchmarkMode),
        cl::Desc("Benchmarking mode to disable keypress event handling")
        ));

    add_cmdline_option(cl::makeOption<std::string&>(
        cl::Parser<>(), "o",
        cl::ArgName("string"),
        cl::ArgOptional,
        cl::init(cmd.sOutputFile),
        cl::Desc("Output tlp file")
        ));

    add_cmdline_option(cl::makeOption<LayouterMode&>({
            {"naive",         LayouterMode::Naive,          "Naive Implementation"},
            { "rtx",          LayouterMode::RTX,          "RTX Mode"             },
            { "lbvh",         LayouterMode::LBVH,          "LBVH Mode"              },
        },
        "mode",
        cl::ArgOptional,
        cl::init(cmd.layouterMode),
        cl::Desc("Select graph layout mode")
        ));


    add_cmdline_option(cl::makeOption<eInputSource&>({
            {"artificial",         eInputSource::ARTIFICIAL,          "Artificial Graph Generation"},
            { "tree",          eInputSource::TREE,          "Tree Generation"             },
            { "file",         eInputSource::FILEINPUT,          "File Input"              },
        },
        "dt",
        cl::ArgOptional,
        cl::init(cmd.inputSource),
        cl::Desc("Select data generation mode")
        ));


    add_cmdline_option(cl::makeOption<unsigned&>(
        cl::Parser<>(), "refit_after",
        cl::ArgOptional,
        cl::init(cmd.rebuildRTXAccelAfter),
        cl::Desc("Refit RTX BVH after N iterations")
        ));

   //Artificial
    add_cmdline_option(cl::makeOption<int&>(cl::Parser<>(), "C", cl::ArgName("int"),
        cl::ArgOptional,cl::init(cmd.artificialOptions.Clusters),cl::Desc("Clusters")
        ));

    add_cmdline_option(cl::makeOption<int&>(cl::Parser<>(), "npc", cl::ArgName("int"),
        cl::ArgOptional,cl::init(cmd.artificialOptions.NodesPerCluster),cl::Desc("Nodes per Clusters")
        ));

    add_cmdline_option(cl::makeOption<int&>(cl::Parser<>(), "epc", cl::ArgName("int"),
        cl::ArgOptional,cl::init(cmd.artificialOptions.EdgesPerCluster),cl::Desc("Edges per Clusters")
        ));

    add_cmdline_option(cl::makeOption<bool&>(cl::Parser<>(), "connected", cl::ArgName("bool"),
        cl::ArgOptional,cl::init(cmd.artificialOptions.Connected),cl::Desc("Generate connected graph")
        ));
 
    add_cmdline_option(cl::makeOption<int&>(cl::Parser<>(), "trDepth", cl::ArgName("int"),
        cl::ArgOptional,cl::init(cmd.treeOptions.depth),cl::Desc("Tree data generation depth")
        ));

    add_cmdline_option(cl::makeOption<int&>(cl::Parser<>(), "trDegree", cl::ArgName("int"),
        cl::ArgOptional,cl::init(cmd.treeOptions.degree),cl::Desc("Tree data generation degree")
        ));

    add_cmdline_option(cl::makeOption<std::set<std::string>&>(
        cl::Parser<>(),
        "files",
        cl::Desc("A list of input files"),
        cl::Positional,
        cl::ZeroOrMore,
        cl::init(cmd.inputFiles)
        ));

}


int main(int argc, char** argv)
{
    renderer rend;
	
    try
    {
        rend.init(argc, argv);
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    rend.active = rend.cmd.benchmarkMode;

    if (!rend.cmd.inputFiles.empty())
        rend.cmd.inputSource = eInputSource::FILEINPUT;


    std::thread thrd([&](){
        if (!rend.cmd.initial_camera.empty()) {
            rend.load_camera(rend.cmd.initial_camera);
        }
        for(int i=0; i < rend.cmd.loadRepetitions;i++) {

            switch(rend.cmd.inputSource) {
                case eInputSource::ARTIFICIAL:
                    rend.graph.loadArtificial(rend.cmd.artificialOptions.Clusters,
                                              rend.cmd.artificialOptions.NodesPerCluster,
                                              rend.cmd.artificialOptions.EdgesPerCluster,
                                              rend.cmd.artificialOptions.Connected);
                break;

                case eInputSource::TREE:
                    rend.graph.loadCompleteTree(rend.cmd.treeOptions.depth, rend.cmd.treeOptions.degree);
                break;

                case eInputSource::FILEINPUT:
                    for(auto f : rend.cmd.inputFiles) {
                        std::cout << "Loading file <" << f << ">\n";
                        // TODO: make this more general
                        boost::filesystem::path p(f);
                        if (p.extension().string() == ".csv")
                            rend.graph.loadGephiCSV(f);
                        else
                            rend.graph.loadDeezerCSV(f); // TODO: deezer files actually also end on .csv (...)
                    }
                break;

            }
        }

        Layouter l(rend.graph, rend.cmd.layouterMode, rend.cmd.rebuildRTXAccelAfter);
        {
            std::unique_lock<std::mutex> l(rend.mtx);
            rend.imageUpdated = true;
            Bounds bounds = rend.graph.getBounds();
            aabb bbox{{bounds.min.x,bounds.min.y,0.f},
                      {bounds.max.x,bounds.max.y,1.f}};
            rend.cam.view_all(bbox);
        }

#if VERBOSE
	    rend.graph.printStatistics(true);
#endif

        for (int noIterations=0; ; )
        {
            bool active = false;

            {
                std::unique_lock<std::mutex> l(rend.mtx);

                if (rend.doQuit)
                    break;

                if (!rend.active)
                    rend.total_time.reset();
                active = rend.active;
            }

            if (active)
            {
                l.iterate();
                {
                    std::unique_lock<std::mutex> l(rend.mtx);
                    rend.imageUpdated = true;
                } 

                if (noIterations==rend.cmd.maxIterations)
                {
                    std::cout << "Total time for " << noIterations << " iterations: " << rend.total_time.elapsed() << " sec.\n";
                    if (rend.cmd.sOutputFile.compare(""))
                    {
                        std::cout << "Writing file <" << rend.cmd.sOutputFile << "> ... ";
                        rend.graph.saveTLP(rend.cmd.sOutputFile);
                        std::cout << "done" << std::endl;
                    }

                    {
                        std::unique_lock<std::mutex> l(rend.mtx);
                        rend.active = false;
                        rend.doQuit = true;
                    }
                    break;
                }
                ++noIterations;
            }
        }
    });

    rend.cam.set_viewport(0, 0, rend.width(), rend.height());
    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), 1.f, 0.001f, 1000.0f);
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
    thrd.join();

    return 0;
}
