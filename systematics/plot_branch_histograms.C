// Summary: Draw configured scalar or C-array ROOT branches from a TChain.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "TBranch.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TFile.h"
#include "TH1D.h"
#include "TLeaf.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

using namespace std;

namespace {

static const vector<string> INPUT_FILES = {
    "dataset/data/example.root"
};

static const string TREE_NAME = "Events";
static const string OUTPUT_DIR = "systematics/branch_histograms";
static const string OUTPUT_ROOT_FILE = "histograms.root";
static const int DEFAULT_BINS = 100;
static const double LOG_AXIS_MIN = 0.1;

struct BranchPlotConfig {
    string name;
    double xmin = 0.;
    double xmax = 1.;
    bool logx = false;
    int bins = DEFAULT_BINS;
};

static const vector<BranchPlotConfig> BRANCHES = {
    // Edit these examples for the ROOT files being checked.
    {"nScoutingFatPFJetRecluster", -0.5, 10.5, false},
    {"ScoutingFatPFJetRecluster_pt", LOG_AXIS_MIN, 2000., true}
};

enum class NumericType {
    Float,
    Double,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    Long64,
    ULong64,
    Char,
    UChar,
    Bool
};

struct NumericBuffer {
    NumericType type = NumericType::Float;
    size_t size = 1;

    vector<Float_t> floats;
    vector<Double_t> doubles;
    vector<Short_t> shorts;
    vector<UShort_t> ushorts;
    vector<Int_t> ints;
    vector<UInt_t> uints;
    vector<Long_t> longs;
    vector<ULong_t> ulongs;
    vector<Long64_t> long64s;
    vector<ULong64_t> ulong64s;
    vector<Char_t> chars;
    vector<UChar_t> uchars;
    unique_ptr<Bool_t[]> bools;

    void allocate(NumericType t, size_t n) {
        type = t;
        size = max<size_t>(1, n);

        floats.clear();
        doubles.clear();
        shorts.clear();
        ushorts.clear();
        ints.clear();
        uints.clear();
        longs.clear();
        ulongs.clear();
        long64s.clear();
        ulong64s.clear();
        chars.clear();
        uchars.clear();
        bools.reset();

        if (type == NumericType::Float) {
            floats.assign(size, 0.f);
        } else if (type == NumericType::Double) {
            doubles.assign(size, 0.);
        } else if (type == NumericType::Short) {
            shorts.assign(size, 0);
        } else if (type == NumericType::UShort) {
            ushorts.assign(size, 0);
        } else if (type == NumericType::Int) {
            ints.assign(size, 0);
        } else if (type == NumericType::UInt) {
            uints.assign(size, 0);
        } else if (type == NumericType::Long) {
            longs.assign(size, 0);
        } else if (type == NumericType::ULong) {
            ulongs.assign(size, 0);
        } else if (type == NumericType::Long64) {
            long64s.assign(size, 0);
        } else if (type == NumericType::ULong64) {
            ulong64s.assign(size, 0);
        } else if (type == NumericType::Char) {
            chars.assign(size, 0);
        } else if (type == NumericType::UChar) {
            uchars.assign(size, 0);
        } else {
            bools.reset(new Bool_t[size]());
        }
    }

    void* address() {
        if (type == NumericType::Float) {
            return floats.data();
        }
        if (type == NumericType::Double) {
            return doubles.data();
        }
        if (type == NumericType::Short) {
            return shorts.data();
        }
        if (type == NumericType::UShort) {
            return ushorts.data();
        }
        if (type == NumericType::Int) {
            return ints.data();
        }
        if (type == NumericType::UInt) {
            return uints.data();
        }
        if (type == NumericType::Long) {
            return longs.data();
        }
        if (type == NumericType::ULong) {
            return ulongs.data();
        }
        if (type == NumericType::Long64) {
            return long64s.data();
        }
        if (type == NumericType::ULong64) {
            return ulong64s.data();
        }
        if (type == NumericType::Char) {
            return chars.data();
        }
        if (type == NumericType::UChar) {
            return uchars.data();
        }
        return bools.get();
    }

    double valueAt(size_t index) const {
        if (index >= size) {
            throw runtime_error("Numeric buffer index is out of range");
        }

        if (type == NumericType::Float) {
            return floats[index];
        }
        if (type == NumericType::Double) {
            return doubles[index];
        }
        if (type == NumericType::Short) {
            return shorts[index];
        }
        if (type == NumericType::UShort) {
            return ushorts[index];
        }
        if (type == NumericType::Int) {
            return ints[index];
        }
        if (type == NumericType::UInt) {
            return uints[index];
        }
        if (type == NumericType::Long) {
            return static_cast<double>(longs[index]);
        }
        if (type == NumericType::ULong) {
            return static_cast<double>(ulongs[index]);
        }
        if (type == NumericType::Long64) {
            return static_cast<double>(long64s[index]);
        }
        if (type == NumericType::ULong64) {
            return static_cast<double>(ulong64s[index]);
        }
        if (type == NumericType::Char) {
            return chars[index];
        }
        if (type == NumericType::UChar) {
            return uchars[index];
        }
        return bools[index] ? 1. : 0.;
    }
};

struct BoundBranch {
    string name;
    string rootTypeName;
    NumericType type = NumericType::Float;
    bool isArray = false;
    string countBranch;
    size_t arraySize = 1;
    NumericBuffer buffer;

    void allocate() {
        buffer.allocate(type, isArray ? arraySize : 1);
    }

    void bind(TTree& tree) {
        if (tree.SetBranchAddress(name.c_str(), buffer.address()) < 0) {
            throw runtime_error("Failed to bind branch: " + name);
        }
    }

    double scalarValue() const {
        return buffer.valueAt(0);
    }

    Long64_t scalarAsLength() const {
        const double value = scalarValue();
        if (!isfinite(value) || value <= 0.) {
            return 0;
        }
        return static_cast<Long64_t>(value);
    }

    double firstArrayValue() const {
        return buffer.valueAt(0);
    }
};

struct PlotRuntime {
    BranchPlotConfig config;
    BoundBranch* valueBranch = nullptr;
    BoundBranch* countBranch = nullptr;
    unique_ptr<TH1D> hist;
};

string stripSpaces(string text) {
    text.erase(remove_if(text.begin(), text.end(),
                         [](unsigned char c) { return isspace(c); }),
               text.end());
    return text;
}

string lower(string text) {
    transform(text.begin(), text.end(), text.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return text;
}

NumericType parseNumericType(const string& rootTypeName, const string& branchName) {
    const string t = lower(stripSpaces(rootTypeName));

    if (t == "float_t" || t == "float" || t == "float16_t") {
        return NumericType::Float;
    }
    if (t == "double_t" || t == "double" || t == "double32_t") {
        return NumericType::Double;
    }
    if (t == "short_t" || t == "short") {
        return NumericType::Short;
    }
    if (t == "ushort_t" || t == "unsignedshort") {
        return NumericType::UShort;
    }
    if (t == "int_t" || t == "int") {
        return NumericType::Int;
    }
    if (t == "uint_t" || t == "unsignedint") {
        return NumericType::UInt;
    }
    if (t == "long_t" || t == "long") {
        return NumericType::Long;
    }
    if (t == "ulong_t" || t == "unsignedlong") {
        return NumericType::ULong;
    }
    if (t == "long64_t" || t == "longlong_t" || t == "longlong") {
        return NumericType::Long64;
    }
    if (t == "ulong64_t" || t == "ulonglong_t" || t == "unsignedlonglong") {
        return NumericType::ULong64;
    }
    if (t == "char_t" || t == "char" || t == "byte_t") {
        return NumericType::Char;
    }
    if (t == "uchar_t" || t == "unsignedchar" || t == "ubyte_t") {
        return NumericType::UChar;
    }
    if (t == "bool_t" || t == "bool") {
        return NumericType::Bool;
    }

    throw runtime_error("Unsupported numeric type '" + rootTypeName +
                        "' for branch '" + branchName + "'");
}

TLeaf* primaryLeaf(TBranch* branch, const string& branchName) {
    TLeaf* leaf = branch->GetLeaf(branchName.c_str());
    if (leaf != nullptr) {
        return leaf;
    }

    const auto* leaves = branch->GetListOfLeaves();
    if (leaves == nullptr || leaves->GetEntries() != 1) {
        throw runtime_error("Branch '" + branchName +
                            "' does not have exactly one numeric leaf");
    }
    return static_cast<TLeaf*>(leaves->At(0));
}

string makeUniquePath(const string& requestedPath) {
    if (gSystem->AccessPathName(requestedPath.c_str())) {
        return requestedPath;
    }

    const size_t slash = requestedPath.find_last_of('/');
    const size_t dot = requestedPath.find_last_of('.');
    const bool hasExtension = (dot != string::npos && (slash == string::npos || dot > slash));
    const string stem = hasExtension ? requestedPath.substr(0, dot) : requestedPath;
    const string ext = hasExtension ? requestedPath.substr(dot) : "";

    for (int index = 1; index < 10000; ++index) {
        const string candidate = stem + "_" + to_string(index) + ext;
        if (gSystem->AccessPathName(candidate.c_str())) {
            return candidate;
        }
    }

    throw runtime_error("Could not find a free output path for " + requestedPath);
}

string safeName(string text) {
    for (char& c : text) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!isalnum(uc) && c != '_') {
            c = '_';
        }
    }
    return text;
}

double effectiveXMin(const BranchPlotConfig& config) {
    return config.logx ? max(config.xmin, LOG_AXIS_MIN) : config.xmin;
}

void validatePlotConfig(const BranchPlotConfig& config) {
    if (config.name.empty()) {
        throw runtime_error("Configured branch name is empty");
    }
    if (config.bins <= 0) {
        throw runtime_error("Branch '" + config.name + "' has non-positive bin count");
    }
    const double xmin = effectiveXMin(config);
    if (!(config.xmax > xmin)) {
        throw runtime_error("Branch '" + config.name + "' has invalid histogram range");
    }
}

BoundBranch inspectBranch(TTree& tree, const string& branchName) {
    TBranch* branch = tree.GetBranch(branchName.c_str());
    if (branch == nullptr) {
        throw runtime_error("Missing branch: " + branchName);
    }

    const char* className = branch->GetClassName();
    if (className != nullptr && className[0] != '\0') {
        throw runtime_error("Branch '" + branchName +
                            "' is an object/vector branch, not a C array or scalar branch");
    }

    TLeaf* leaf = primaryLeaf(branch, branchName);
    const string typeName = leaf->GetTypeName();
    TLeaf* countLeaf = leaf->GetLeafCount();

    BoundBranch out;
    out.name = branchName;
    out.rootTypeName = typeName;
    out.type = parseNumericType(typeName, branchName);
    out.isArray = (countLeaf != nullptr || leaf->GetLenStatic() > 1);

    if (countLeaf != nullptr) {
        out.countBranch = countLeaf->GetName();
        Long64_t observedMax = static_cast<Long64_t>(ceil(tree.GetMaximum(out.countBranch.c_str())));
        if (observedMax < 0) {
            observedMax = 0;
        }
        out.arraySize = max<Long64_t>(1, observedMax);
    } else if (out.isArray) {
        out.arraySize = max(1, leaf->GetLenStatic());
    }

    return out;
}

BoundBranch& ensureBoundBranch(TTree& tree,
                               map<string, BoundBranch>& boundBranches,
                               const string& branchName) {
    auto it = boundBranches.find(branchName);
    if (it != boundBranches.end()) {
        return it->second;
    }

    BoundBranch branch = inspectBranch(tree, branchName);
    auto inserted = boundBranches.emplace(branchName, std::move(branch));
    return inserted.first->second;
}

vector<PlotRuntime> buildPlots(TTree& tree, map<string, BoundBranch>& boundBranches) {
    vector<PlotRuntime> plots;
    plots.reserve(BRANCHES.size());

    for (const BranchPlotConfig& config : BRANCHES) {
        validatePlotConfig(config);

        BoundBranch& valueBranch = ensureBoundBranch(tree, boundBranches, config.name);
        BoundBranch* countBranch = nullptr;
        if (valueBranch.isArray && !valueBranch.countBranch.empty()) {
            countBranch = &ensureBoundBranch(tree, boundBranches, valueBranch.countBranch);
            if (countBranch->isArray) {
                throw runtime_error("Count branch '" + valueBranch.countBranch +
                                    "' for array branch '" + valueBranch.name +
                                    "' is not scalar");
            }
        }

        const string histName = "h_" + safeName(config.name);
        const double xmin = effectiveXMin(config);
        unique_ptr<TH1D> hist(new TH1D(histName.c_str(), "", config.bins, xmin, config.xmax));
        hist->SetDirectory(nullptr);
        hist->SetLineWidth(2);
        hist->SetTitle((config.name + ";" + config.name + ";Events").c_str());

        plots.push_back(PlotRuntime{config, &valueBranch, countBranch, std::move(hist)});
    }

    return plots;
}

void configureBranches(TTree& tree, map<string, BoundBranch>& boundBranches) {
    tree.SetBranchStatus("*", 0);
    for (const auto& item : boundBranches) {
        tree.SetBranchStatus(item.first.c_str(), 1);
    }
    tree.SetCacheSize(50 * 1024 * 1024);
    for (const auto& item : boundBranches) {
        tree.AddBranchToCache(item.first.c_str(), true);
    }

    for (auto& item : boundBranches) {
        item.second.allocate();
        item.second.bind(tree);
    }
}

void fillPlots(TTree& tree, vector<PlotRuntime>& plots) {
    const Long64_t nEntries = tree.GetEntries();
    cout << "Total entries = " << nEntries << endl;

    for (Long64_t entry = 0; entry < nEntries; ++entry) {
        tree.GetEntry(entry);

        if (entry % 1000000 == 0) {
            const double percent = (nEntries > 0) ? 100. * static_cast<double>(entry + 1) / nEntries : 100.;
            cout << "\rProcessing entry " << (entry + 1) << " / " << nEntries
                 << " (" << percent << "%)" << flush;
        }

        for (PlotRuntime& plot : plots) {
            double value = 0.;
            if (plot.valueBranch->isArray) {
                const Long64_t count = plot.countBranch != nullptr
                    ? plot.countBranch->scalarAsLength()
                    : static_cast<Long64_t>(plot.valueBranch->arraySize);
                if (count <= 0) {
                    continue;
                }
                value = plot.valueBranch->firstArrayValue();
            } else {
                value = plot.valueBranch->scalarValue();
            }

            if (isfinite(value)) {
                plot.hist->Fill(value);
            }
        }
    }

    if (nEntries > 0) {
        cout << endl;
    }
}

void savePlots(const vector<PlotRuntime>& plots) {
    gSystem->mkdir(OUTPUT_DIR.c_str(), kTRUE);

    const string rootPath = makeUniquePath(OUTPUT_DIR + "/" + OUTPUT_ROOT_FILE);
    TFile output(rootPath.c_str(), "CREATE");
    if (output.IsZombie()) {
        throw runtime_error("Failed to create output ROOT file: " + rootPath);
    }

    for (const PlotRuntime& plot : plots) {
        plot.hist->Write();
    }
    output.Close();

    for (const PlotRuntime& plot : plots) {
        TCanvas canvas(("c_" + safeName(plot.config.name)).c_str(), "", 800, 700);
        canvas.SetMargin(0.12, 0.04, 0.12, 0.08);
        canvas.SetLogy(true);
        if (plot.config.logx) {
            canvas.SetLogx(true);
        }

        plot.hist->SetMinimum(LOG_AXIS_MIN);
        if (plot.hist->GetMaximum() > 0.) {
            plot.hist->SetMaximum(plot.hist->GetMaximum() * 10.);
        } else {
            plot.hist->SetMaximum(1.);
        }
        plot.hist->Draw("hist");

        const string pdfPath = makeUniquePath(OUTPUT_DIR + "/" + safeName(plot.config.name) + ".pdf");
        canvas.SaveAs(pdfPath.c_str());
        cout << "Wrote " << pdfPath << endl;
    }

    cout << "Wrote " << rootPath << endl;
}

void printBranchSummary(const vector<PlotRuntime>& plots) {
    for (const PlotRuntime& plot : plots) {
        const BoundBranch& branch = *plot.valueBranch;
        cout << "Branch " << branch.name << ": type = " << branch.rootTypeName;
        if (branch.isArray) {
            cout << ", C array";
            if (!branch.countBranch.empty()) {
                cout << ", count branch = " << branch.countBranch;
            }
            cout << ", buffer size = " << branch.arraySize
                 << ", filling index 0";
        } else {
            cout << ", scalar";
        }
        cout << endl;
    }
}

}  // namespace

int plot_branch_histograms() {
    try {
        if (INPUT_FILES.empty()) {
            throw runtime_error("INPUT_FILES is empty");
        }
        if (BRANCHES.empty()) {
            throw runtime_error("BRANCHES is empty");
        }

        gROOT->SetBatch(kTRUE);
        gStyle->SetOptStat(0);

        TChain chain(TREE_NAME.c_str());
        for (const string& inputFile : INPUT_FILES) {
            cout << "Adding file: " << inputFile << endl;
            chain.Add(inputFile.c_str());
        }

        if (chain.GetEntries() > 0) {
            chain.LoadTree(0);
        }

        map<string, BoundBranch> boundBranches;
        vector<PlotRuntime> plots = buildPlots(chain, boundBranches);
        configureBranches(chain, boundBranches);
        printBranchSummary(plots);
        fillPlots(chain, plots);
        savePlots(plots);

        chain.ResetBranchAddresses();
    } catch (const exception& ex) {
        cerr << "plot_branch_histograms error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}

#ifndef __CLING__
int main() {
    return plot_branch_histograms();
}
#endif
