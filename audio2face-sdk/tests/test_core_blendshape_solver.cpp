// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
#include <random>
#include <future>
#include <type_traits>

#include <gtest/gtest.h>

#include "audio2face/internal/job_runner.h"
#include "audio2face/internal/blendshape_solver.h"
#include "audio2face/internal/blendshape_solver_gpu.h"
#include "audio2face/internal/macros.h"
#include "audio2face/internal/eigen_utils.h"
#include "audio2face/internal/parse_helper.h"
#include "audio2x/internal/unique_ptr.h"
#include "audio2x/cuda_utils.h"
#include "utils.h"

#include <cuda_runtime_api.h>

using namespace Eigen;
using namespace nva2f;

namespace {

constexpr std::array<const char*, 3> BS_DATA_PATHS = {
    TEST_DATA_DIR "_data/audio2face-models/audio2face-3d-v2.3-mark/bs_skin.npz",
    TEST_DATA_DIR "_data/audio2face-models/audio2face-3d-v2.3-mark/bs_tongue.npz",
    TEST_DATA_DIR "_data/audio2x-sdk-test-data/audio2face-sdk-test-data/bs_mark180.npz"
};
constexpr std::array<std::array<std::pair<const char*, int>, 3>, 3> POSE_NAME_IDX = {{
    {{
        {"eyeBlinkLeft", 0},
        {"mouthLowerDownRight", 38},
        {"tongueOut", 51}
    }},
    {{
        {"tongueTipUp", 0},
        {"tongueRollLeft", 6},
        {"tongueNarrow", 15}
    }},
    {{
        {"neutral_MH_Head_jaw_TX_E1", 0},
        {"neutral_MH_Head_R_mouth_corner_TY_E2", 123},
        {"neutral_MH_Head_R_eye_TY_E2", 179}
    }}
}};

static_assert(std::tuple_size<decltype(BS_DATA_PATHS)>::value == std::tuple_size<decltype(POSE_NAME_IDX)>::value,
              "Arrays must have the same size!");

constexpr float MAETolerance = 0.2f; // Set based on the results from sample-a2f-metrics

Eigen::VectorXf getSampleWeights(int poseSize) {
    // randomly set some weights
    Eigen::VectorXf weights(poseSize);
    weights.setZero();
    // fixed data for reproducibile test
    switch (weights.size()) {
    case 16:
        weights(2) = 0.156f;
        weights(3) = 0.670f;
        weights(5) = 0.479f;
        weights(7) = 0.074f;
        weights(11) = 0.076f;
        weights(13) = 0.121f;
        break;
    case 52:
        weights(14) = 0.156f;
        weights(19) = 0.670f;
        weights(20) = 0.479f;
        weights(23) = 0.074f;
        weights(24) = 0.076f;
        weights(39) = 0.121f;
        weights(40) = 0.120f;
        break;
    case 180:
        weights(14) = 0.156f;
        weights(25) = 0.670f;
        weights(47) = 0.479f;
        weights(73) = 0.074f;
        weights(125) = 0.076f;
        weights(150) = 0.121f;
        weights(163) = 0.120f;
        break;
    default:
        throw std::runtime_error("Test data size has changed!");
    }
    return weights;
}

std::vector<float> computeVertexDistances(const Eigen::VectorXf& inferred, const Eigen::VectorXf& solved){
    int numVertices = static_cast<int>(inferred.size() / 3);
    std::vector<float> distances(numVertices);

    for (int i = 0; i < numVertices; i++) {
        int idx = 3 * i;
        Eigen::Vector3f vertexInferred = inferred.segment<3>(idx);
        Eigen::Vector3f vertexSolved   = solved.segment<3>(idx);
        distances[i] = (vertexInferred - vertexSolved).norm();  // Euclidean distance for each vertex
    }

    return distances;
}

float MAE(const std::vector<float>& distances){
    float sumAbs = 0.0;
    for (float dist : distances) {
        sumAbs += std::abs(dist);
    }
    return sumAbs / distances.size();
}

} // Anonymouse Namespace

// This is a workaround to specify path and type
template <typename T, int bsDataIdx>
struct TestParams {
    using Implementation = T;
    static constexpr auto DataIdx = bsDataIdx;
    static const char* getBsDataPath() { return BS_DATA_PATHS[bsDataIdx]; }
    static auto& getPoseNameIdxMapping() { return POSE_NAME_IDX[bsDataIdx]; }
    static int getBsDataIdx() { return bsDataIdx; }
};

template <typename ParamType>
class TestBlendshapeSolver : public ::testing::Test {
protected:
    using SolverImplementation = typename ParamType::Implementation;
    static constexpr auto SolverDataIdx = ParamType::DataIdx;

    cudaStream_t cudaStream;
    nva2x::UniquePtr<IBlendshapeSolver> solver;
    nva2x::UniquePtr<nva2f::IJobRunner> jobRunner;
    nva2x::DeviceTensorFloat targetPoseDevice;

    void SetUp() override {
        cudaStreamCreate(&cudaStream);
        jobRunner = nva2x::ToUniquePtr(CreateThreadPoolJobRunner_INTERNAL(1));
        solver = nva2x::ToUniquePtr(new SolverImplementation());
        BlendshapeSolverDataOwner bsData;
        bsData.Init(ParamType::getBsDataPath());
        solver->SetBlendshapeData(bsData.GetBlendshapeSolverDataView());
        targetPoseDevice.Allocate(solver->PoseSize());
        solver->SetCudaStream(cudaStream);
        solver->SetJobRunner(jobRunner.get());
        std::vector<float> multipliers(solver->NumBlendshapePoses(), 1.0f);
        std::vector<float> offsets(solver->NumBlendshapePoses(), 0.0f);
        solver->SetMultipliers(nva2x::ToConstView(multipliers));
        solver->SetOffsets(nva2x::ToConstView(offsets));
        BlendshapeSolverParams params;
        params.L1Reg = 0.5f;
        params.L2Reg = 0.7f;
        params.SymmetryReg = 0.0f;
        params.TemporalReg = 0.0f;
        params.templateBBSize = 54.7f;
        params.tolerance = 1e-10f;
        solver->SetParameters(params);
        solver->Prepare();

        cudaError_t status = cudaStreamSynchronize(cudaStream);
        if (status != cudaSuccess) {
            std::cout << "CUDA error: " << cudaGetErrorString(status) << std::endl;
        }
    }

    void TearDown() override {
        cudaStreamDestroy(cudaStream);
    }
};

// Helper to check if TypeParam is TestParams<BlendshapeSolver, N>
template <typename T>
constexpr bool IsBlendshapeSolverType = false;

template <int N>
constexpr bool IsBlendshapeSolverType<TestParams<BlendshapeSolver, N>> = true;

using TestTypes = ::testing::Types<
    TestParams<BlendshapeSolver, 0>,
    TestParams<BlendshapeSolver, 1>,
    TestParams<BlendshapeSolver, 2>,
    TestParams<BlendshapeSolverGPU, 0>,
    TestParams<BlendshapeSolverGPU, 1>,
    TestParams<BlendshapeSolverGPU, 2>
>;
TYPED_TEST_SUITE(TestBlendshapeSolver, TestTypes);

TYPED_TEST(TestBlendshapeSolver, TestBlendshapeSolveAsync) {
    Eigen::VectorXf sampleWeights = getSampleWeights(this->solver->NumBlendshapePoses());

    Eigen::VectorXf targetPose(this->solver->PoseSize());
    this->solver->EvaluatePose(ToConstView(sampleWeights), ToView(targetPose));
    ASSERT_TRUE(!nva2x::CopyHostToDevice(this->targetPoseDevice, ToView(targetPose)));

    Eigen::VectorXf solvedWeights = Eigen::VectorXf::Zero(sampleWeights.size());

    if constexpr (IsBlendshapeSolverType<TypeParam>) {
        std::promise<std::error_code> promise;
        std::future<std::error_code> future = promise.get_future();
        ASSERT_TRUE(!this->solver->SolveAsync(this->targetPoseDevice, ToView(solvedWeights), [](void* p, std::error_code error) {
            reinterpret_cast<std::promise<std::error_code>*>(p)->set_value(error);
        }, &promise));
        // just in case there's something wrong we can still finish the unit test
        auto futureStatus = future.wait_for(std::chrono::seconds(3));
        ASSERT_EQ(futureStatus, std::future_status::ready) << "Blendshape solve did not complete in 3 seconds";
        ASSERT_TRUE(!future.get());
    } else {
        nva2x::DeviceTensorFloat solvedWeightsDevice;
        ASSERT_TRUE(!solvedWeightsDevice.Allocate(this->solver->NumBlendshapePoses()));
        ASSERT_TRUE(!this->solver->SolveAsync(this->targetPoseDevice, solvedWeightsDevice));
        ASSERT_TRUE(!nva2x::CopyDeviceToHost(ToView(solvedWeights), solvedWeightsDevice, this->cudaStream));
    }

    Eigen::VectorXf solvedPose(this->solver->PoseSize());
    this->solver->EvaluatePose(ToConstView(solvedWeights), ToView(solvedPose));

    std::vector<float> distances = computeVertexDistances(targetPose, solvedPose);
    ASSERT_LE(MAE(distances), MAETolerance);
}

TYPED_TEST(TestBlendshapeSolver, TestBlendshapeSolve) {
    Eigen::VectorXf sampleWeights = getSampleWeights(this->solver->NumBlendshapePoses());

    Eigen::VectorXf targetPose(this->solver->PoseSize());
    this->solver->EvaluatePose(ToConstView(sampleWeights), ToView(targetPose));
    ASSERT_TRUE(!nva2x::CopyHostToDevice(this->targetPoseDevice, ToView(targetPose)));

    Eigen::VectorXf solvedWeights = Eigen::VectorXf::Zero(sampleWeights.size());

    ASSERT_TRUE(!this->solver->Solve(this->targetPoseDevice, ToView(solvedWeights)));

    Eigen::VectorXf solvedPose(this->solver->PoseSize());
    this->solver->EvaluatePose(ToConstView(solvedWeights), ToView(solvedPose));

    std::vector<float> distances = computeVertexDistances(targetPose, solvedPose);
    ASSERT_LE(MAE(distances), MAETolerance);
}

TYPED_TEST(TestBlendshapeSolver, TestSettersInvalidatePreparedState) {
    auto testInvalidatePrepare = [this]() {
        // call this function after calling a setter that invalidates the prepared state
        Eigen::VectorXf solvedWeights = Eigen::VectorXf::Zero(this->solver->NumBlendshapePoses());
        // the first call right after setting the parameters should fail
        ASSERT_TRUE(this->solver->Solve(this->targetPoseDevice, ToView(solvedWeights)));

        // after preparing again, the solve should work
        ASSERT_TRUE(!this->solver->Prepare());
        ASSERT_TRUE(!this->solver->Solve(this->targetPoseDevice, ToView(solvedWeights)));
    };

    std::vector<int> activePoses(this->solver->NumBlendshapePoses());
    ASSERT_TRUE(!this->solver->GetActivePoses(activePoses.data(), activePoses.size()));
    ASSERT_TRUE(!this->solver->SetActivePoses(activePoses.data(), activePoses.size()));
    testInvalidatePrepare();

    std::vector<int> cancelPoses(this->solver->NumBlendshapePoses());
    ASSERT_TRUE(!this->solver->GetCancelPoses(cancelPoses.data(), cancelPoses.size()));
    ASSERT_TRUE(!this->solver->SetCancelPoses(cancelPoses.data(), cancelPoses.size()));
    testInvalidatePrepare();

    std::vector<int> symmetryPoses(this->solver->NumBlendshapePoses());
    ASSERT_TRUE(!this->solver->GetSymmetryPoses(symmetryPoses.data(), symmetryPoses.size()));
    ASSERT_TRUE(!this->solver->SetSymmetryPoses(symmetryPoses.data(), symmetryPoses.size()));
    testInvalidatePrepare();

    BlendshapeSolverParams params = this->solver->GetParameters();
    ASSERT_TRUE(!this->solver->SetParameters(params));
    testInvalidatePrepare();
}

TYPED_TEST(TestBlendshapeSolver, TestMultipliersAndOffsets) {
    const float kTargetOffset = 0.314159f;
    std::vector<float> multipliers(this->solver->NumBlendshapePoses(), 0.0f);
    std::vector<float> offsets(this->solver->NumBlendshapePoses(), kTargetOffset);

    Eigen::VectorXf solvedWeights = Eigen::VectorXf::Zero(this->solver->NumBlendshapePoses());
    // Call solve to make sure the prepared state is valid.
    ASSERT_TRUE(!this->solver->Solve(this->targetPoseDevice, ToView(solvedWeights)));

    // Also test that the multipliers and offsets doesn't require Prepare() to be called
    ASSERT_TRUE(!this->solver->SetMultipliers(nva2x::ToConstView(multipliers)));
    ASSERT_TRUE(!this->solver->SetOffsets(nva2x::ToConstView(offsets)));
    ASSERT_TRUE(!this->solver->Solve(this->targetPoseDevice, ToView(solvedWeights)));

    for (int i = 0; i < this->solver->NumBlendshapePoses(); ++i) {
        ASSERT_EQ(solvedWeights[i], kTargetOffset);
    }

    // fill offsets with another value
    const float kAnotherOffset = 0.123456f;
    for (int i = 0; i < this->solver->NumBlendshapePoses(); ++i) {
        offsets[i] = kAnotherOffset;
    }
    // we also don't need to call Prepare() this time
    ASSERT_TRUE(!this->solver->SetOffsets(nva2x::ToConstView(offsets)));
    ASSERT_TRUE(!this->solver->Solve(this->targetPoseDevice, ToView(solvedWeights)));

    for (int i = 0; i < this->solver->NumBlendshapePoses(); ++i) {
        ASSERT_EQ(solvedWeights[i], kAnotherOffset);
    }

}

TYPED_TEST(TestBlendshapeSolver, TestGetPoseNames) {
    auto numPoses = this->solver->NumBlendshapePoses();

    const auto& poseNameIdxMapping = TypeParam::getPoseNameIdxMapping();

    for(int i=0;i<poseNameIdxMapping.size();++i) {
        const char* poseName = this->solver->GetPoseName(poseNameIdxMapping[i].second);
        EXPECT_TRUE(poseName != nullptr);
        EXPECT_STREQ(poseNameIdxMapping[i].first, poseName);
    }

    for(int i=0;i<numPoses;++i) {
        EXPECT_TRUE(this->solver->GetPoseName(i) != nullptr);
    }

    EXPECT_TRUE(this->solver->GetPoseName(numPoses+1) == nullptr);
    EXPECT_TRUE(this->solver->GetPoseName(-1) == nullptr);
}

TYPED_TEST(TestBlendshapeSolver, TestGetterAndSetterWithPoseName) {
    std::array<const char*, 2> invalidPoseNames = {"neutral", nullptr};
    for(auto& invalidPoseName : invalidPoseNames) {
        int outInt;
        float outFloat;
        ASSERT_FALSE(!this->solver->SetActivePose(invalidPoseName, 1));
        ASSERT_FALSE(!this->solver->GetActivePose(invalidPoseName, outInt));
        ASSERT_FALSE(!this->solver->SetCancelPose(invalidPoseName, 1));
        ASSERT_FALSE(!this->solver->GetCancelPose(invalidPoseName, outInt));
        ASSERT_FALSE(!this->solver->SetSymmetryPose(invalidPoseName, 1));
        ASSERT_FALSE(!this->solver->GetSymmetryPose(invalidPoseName, outInt));
        ASSERT_FALSE(!this->solver->SetMultiplier(invalidPoseName, 1));
        ASSERT_FALSE(!this->solver->GetMultiplier(invalidPoseName, outFloat));
        ASSERT_FALSE(!this->solver->SetOffset(invalidPoseName, 1));
        ASSERT_FALSE(!this->solver->GetOffset(invalidPoseName, outFloat));
    }

    const auto& poseNameIdxMapping = TypeParam::getPoseNameIdxMapping();

    for(int i=0;i<poseNameIdxMapping.size();++i) {
        const char* poseName = poseNameIdxMapping[i].first;
        int poseIdx = poseNameIdxMapping[i].second;
        auto testInt = [this, poseIdx, poseName](
            std::error_code (IBlendshapeSolver::*setAll)(const int*, size_t),
            std::error_code (IBlendshapeSolver::*getAll)(int*, size_t),
            std::error_code (IBlendshapeSolver::*setSingle)(const char* poseName, const int val),
            std::error_code (IBlendshapeSolver::*getSingle)(const char* poseName, int& val)) {
            std::vector<int> origData(this->solver->NumBlendshapePoses(), -1);
            ASSERT_TRUE(!(this->solver.get()->*getAll)(origData.data(), origData.size()));
            int outInt;
            ASSERT_TRUE(!(this->solver.get()->*getSingle)(poseName, outInt));
            ASSERT_EQ(outInt, origData[poseIdx]);
            ASSERT_TRUE(!(this->solver.get()->*setSingle)(poseName, outInt+1));
            ASSERT_TRUE(!(this->solver.get()->*getSingle)(poseName, outInt));
            ASSERT_NE(outInt, origData[poseIdx]);
            ASSERT_TRUE(!(this->solver.get()->*setAll)(origData.data(), origData.size()));
        };
        testInt(&IBlendshapeSolver::SetActivePoses, &IBlendshapeSolver::GetActivePoses,
            &IBlendshapeSolver::SetActivePose, &IBlendshapeSolver::GetActivePose);
        testInt(&IBlendshapeSolver::SetCancelPoses, &IBlendshapeSolver::GetCancelPoses,
            &IBlendshapeSolver::SetCancelPose, &IBlendshapeSolver::GetCancelPose);
        testInt(&IBlendshapeSolver::SetSymmetryPoses, &IBlendshapeSolver::GetSymmetryPoses,
            &IBlendshapeSolver::SetSymmetryPose, &IBlendshapeSolver::GetSymmetryPose);

        auto testFloat = [this, poseIdx, poseName](
            std::error_code (IBlendshapeSolver::*setAll)(nva2x::HostTensorFloatConstView),
            std::error_code (IBlendshapeSolver::*getAll)(nva2x::HostTensorFloatView),
            std::error_code (IBlendshapeSolver::*setSingle)(const char* poseName, const float val),
            std::error_code (IBlendshapeSolver::*getSingle)(const char* poseName, float& val)) {
            std::vector<float> origData(this->solver->NumBlendshapePoses(), -1);
            nva2x::HostTensorFloatView origDataView(origData.data(), origData.size());
            nva2x::HostTensorFloatConstView origDataConstView(origData.data(), origData.size());
            ASSERT_TRUE(!(this->solver.get()->*getAll)(origDataView));
            float outFloat;
            ASSERT_TRUE(!(this->solver.get()->*getSingle)(poseName, outFloat));
            ASSERT_EQ(outFloat, origData[poseIdx]);
            ASSERT_TRUE(!(this->solver.get()->*setSingle)(poseName, outFloat+1));
            ASSERT_TRUE(!(this->solver.get()->*getSingle)(poseName, outFloat));
            ASSERT_NE(outFloat, origData[poseIdx]);
            ASSERT_TRUE(!(this->solver.get()->*setAll)(origDataConstView));
        };
        testFloat(&IBlendshapeSolver::SetMultipliers, &IBlendshapeSolver::GetMultipliers,
            &IBlendshapeSolver::SetMultiplier, &IBlendshapeSolver::GetMultiplier);
        testFloat(&IBlendshapeSolver::SetOffsets, &IBlendshapeSolver::GetOffsets,
            &IBlendshapeSolver::SetOffset, &IBlendshapeSolver::GetOffset);
    }

}

// This test is only for CPU solver, and we run it only for the first dataset.
using TestBlendshapeSolver0 = TestBlendshapeSolver<TestParams<BlendshapeSolver, 0>>;
TEST_F(TestBlendshapeSolver0, TestCallbackOrder) {
    const int seed = static_cast<unsigned int>(time(NULL));
    std::cout << "Current srand seed: " << seed << std::endl;
    std::srand(seed); // make random inputs reproducible

    // Assign more threads to the solver.
    jobRunner = nva2x::ToUniquePtr(CreateThreadPoolJobRunner_INTERNAL(8));
    solver->SetJobRunner(jobRunner.get());

    std::mutex mutex;
    std::vector<std::size_t> callbackOrder;

    struct CallbackParam {
        std::mutex& mutex;
        std::vector<std::size_t>& callbackOrder;
        std::size_t callbackIdx;
    };

    auto callback = [](void* p, std::error_code error) {
        auto param = static_cast<CallbackParam*>(p);

        // Sleep a random amount of time to mess with callback order.
        const int sleepTime = std::rand() % 100;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));

        std::lock_guard<std::mutex> lock(param->mutex);
        param->callbackOrder.push_back(param->callbackIdx);

        delete param;
    };

    static constexpr std::size_t kNbSolves = 100;
    std::vector<std::vector<float>> solvedWeights(kNbSolves);
    for (auto& weights : solvedWeights) {
        weights.resize(solver->NumBlendshapePoses(), 0.0f);
    }

    for (std::size_t i = 0; i < kNbSolves; ++i) {
        auto param = new CallbackParam{mutex, callbackOrder, i};
        ASSERT_TRUE(!solver->SolveAsync(targetPoseDevice, nva2x::ToView(solvedWeights[i]), callback, param));
    }

    // Wait for all solves to complete.
    ASSERT_TRUE(!solver->Wait());

    // Check that the callback order is correct.
    for (std::size_t i = 0; i < kNbSolves; ++i) {
        ASSERT_EQ(callbackOrder[i], i);
    }
}
