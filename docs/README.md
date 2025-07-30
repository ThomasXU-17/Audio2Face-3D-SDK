# Audio2X SDK Documentation

## Table of Contents

1. [Overview](#overview)
   - [Key Features](#key-features)
2. [Architecture](#architecture)
   - [Core Design Principles](#core-design-principles)
3. [Components](#components)
   - [Audio2Emotion SDK](#audio2emotion-sdk)
   - [Audio2Face SDK](#audio2face-sdk)
   - [Audio2X Common](#audio2x-common)
4. [Examples](#examples)
5. [Core Concepts](#core-concepts)
   - [Audio Accumulator](#audio-accumulator)
   - [Emotion Accumulator](#emotion-accumulator)
   - [Multi-Track Execution](#multi-track-execution)
   - [Executors](#executors)
     - [Geometry Executors](#geometry-executors)
     - [Blendshape Executors](#blendshape-executors)
   - [Error Reporting](#error-reporting)
     - [Basic Error Checking](#basic-error-checking)
   - [ABI Compatibility](#abi-compatibility)
6. [Models Breakdown](#models-breakdown)
   - [Audio2Emotion](#audio2emotion)
   - [Audio2Face](#audio2face)
     - [Regression-based Animation](#regression-based-animation)
     - [Diffusion-based Animation](#diffusion-based-animation)
     - [Post-processing](#post-processing)
7. [Conclusion](#conclusion)

---

## Overview

The Audio2X SDK is a comprehensive toolkit for fast audio-driven animation and emotion detection. It consists of two main components:

- **Audio2Emotion SDK**: Analyzes audio (speech) input to detect and classify emotional states
- **Audio2Face SDK**: Generates facial animations from audio (speech) input, supporting both regression and diffusion-based approaches

The SDK is designed for high-performance applications requiring fast audio processing and animation generation. It leverages NVIDIA's GPU computing capabilities through CUDA and TensorRT for optimal performance.

### Key Features

- **Faster Than Real-time Processing**: Faster than 60 FPS frame generation
- **Multi-track Support**: Process multiple audio streams simultaneously
- **GPU Acceleration**: Full CUDA/TensorRT integration
- **Flexible Architecture**: Support for both batch and interactive processing modes
- **Cross-platform**: Windows and Linux support

---

## Architecture

The Audio2X SDK provides two main levels of API:

- **Low-level**: Provides fine-grained control over buffer management, inference window progress, inference results post-processing, etc., for advanced use cases.
- **High-level**: Provides turn-key model setup and execution suitable for most use cases.

The low-level API provides many building blocks used to set up and execute a given model, including:

- Common
  - `ICudaStream`: CUDA stream object
  - `IInferenceEngine`: TensorRT execution context
- Audio2Emotion
  - `IMultiTrackPostProcessor`: Post-processes emotion inference results
- Audio2Face
  - `IMultiTrackAnimatorPcaReconstruction`: Handles PCA reconstruction of inference results
  - `IMultiTrackAnimatorSkin`: Handles skin post-processing
  - `IMultiTrackAnimatorTongue`: Handles tongue post-processing
  - `IMultiTrackAnimatorTeeth`: Handles teeth post-processing
  - `IMultiTrackAnimatorEyes`: Handles eyes post-processing
  - `IBlendshapeSolver`: Handles blendshape weights extraction from mesh geometry

These building blocks can be composed to achieve an end-to-end audio processing pipeline, which is what the high-level executor API provides, removing the need to interact with low-level objects.

Multi-track executors support high-throughput processing for offline applications. They handle initialization, execution, and data flow between components to provide final frames using a common API that abstracts the underlying model. Their API allows you to:

1. Accumulate audio to be processed
2. Trigger execution
3. Receive final frame results through a callback mechanism

They allow audio to be processed as it is received in a streaming manner. All audio does not need to be available before execution can start.

Interactive executors are optimized for parameter editing and designed for interactive authoring applications. They expect the whole audio track to be available upfront. They provide two main pieces of functionality:

1. Compute a single frame quickly after parameter editing to provide interactive updates while tweaking values
2. Compute all frames as fast as possible after the final value to visualize continuous animation

### Core Design Principles

1. **Performance First**: Optimized using a GPU-first approach
2. **Model Abstraction**: A common interface should be provided even if underlying models differ
3. **Asynchronicity**: Execution calls should schedule work and return immediately
4. **Efficiency**: No resources (memory, CPU, or GPU cycles) should be wasted on throw-away data or disabled features

---

## Components

### Audio2Emotion SDK

The Audio2Emotion SDK provides emotion detection capabilities from audio input. It uses deep learning models to classify emotional states.

#### Key Components

- **Classifier Models**: Neural networks that analyze audio features to detect emotions
- **Post-processing**: Refines classifier outputs for smoother emotion curves

### Audio2Face SDK

The Audio2Face SDK generates facial animations from audio input using two main approaches:

- **Regression-based model**
- **Diffusion-based model**

Each approach has its implications, which are detailed in the [Models Breakdown](#models-breakdown) section.

Both models output the same facial features:

1. **Skin Geometry**: Skin geometry vertex positions
2. **Tongue Geometry**: Tongue geometry vertex positions
3. **Jaw Transform**: Anchor point positions from which the jaw rigid transform (rotation and translation) can be extracted
4. **Eyes Rotation**: Eyes rotation driven by speech and saccade movement

Geometry results provided by these models can optionally be converted to blendshape weights using a specialized solver.

### Audio2X Common

The Audio2X Common module provides shared infrastructure used by both SDKs:

#### GPU Utilities

- **CUDA Stream Management**: Used for asynchronous GPU operations
- **Tensor Operations**: Device and host tensor allocation and operations
- **Inference Engine**: TensorRT execution context

#### Data Processing

- **Audio Accumulator**: Manages audio buffer accumulation and streaming
- **Emotion Accumulator**: Communication channel for emotion input and output

---

## Examples

See examples in:

- `audio2emotion-sdk/source/samples/`
- `audio2face-sdk/source/samples/`

---

## Core Concepts

### Audio Accumulator

The SDK processes audio using a streaming architecture. It allows sample accumulation in GPU memory and retrieval as required by inference.

Audio is passed as floating-point values to an `IAudioAccumulator` object. Note that the same audio accumulator can be shared between different executors (for example, Audio2Emotion and Audio2Face).

Audio is passed to the accumulator through the `Accumulate()` call. To signal that all audio has been streamed, `Close()` must be called on the accumulator. A closed accumulator is guaranteed not to have new data coming in, which allows executors to read "past the end" (using zero padding) to process the last few frames of the audio track.

Once all the audio has been processed, accumulators can be `Reset()` to accumulate a new audio track.

Accumulators maintain an internal pool of GPU memory buffers and only allocate when the pool is running empty. After `Reset()`, existing buffers are put back in the pool for reuse. To avoid GPU allocation during execution, audio accumulators can pre-allocate memory in advance. It is also possible for accumulators to drop data that has already been processed using `DropSamplesBefore()`. By periodically calling this function as data gets processed, buffers can be reused for new streamed data, making it possible to stream very long audio tracks without endlessly growing GPU memory usage.

### Emotion Accumulator

Emotion accumulators work very similarly to audio accumulators. Emotions are added using the `Accumulate()` call, and `Close()` is used to signal that no additional emotions will be passed.

Unlike audio accumulators, which receive arrays of contiguous samples, emotion accumulators store multi-valued emotions associated with a timestamp. Emotions must be accumulated in increasing timestamp order.

Querying an emotion value at a given timestamp will either copy the value at the exact timestamp if it exists, or interpolate linearly between two emotions if the timestamp falls between two accumulated emotions. Reading before the first emotion returns the first emotion; reading past the last emotion returns the last emotion only if the emotion accumulator is closed. Otherwise, it returns an error since the next accumulated emotion would affect the results of the value queried after the last emotion.

Emotion accumulators are used to abstract how models use emotion input to affect inference. Executors can sample the emotion accumulator regardless of how it was filled. It can contain a single constant value, animated values from DCC animation curves, or values generated by another executor such as Audio2Emotion executors.

### Multi-Track Execution

Audio2Emotion and Audio2Face executors support multi-track execution for more efficient GPU usage. It is easier for GPUs to efficiently process larger workloads than multiple smaller workloads. Therefore, processing multiple audio tracks simultaneously can result in more efficient GPU usage and higher throughput.

Each track has its own audio accumulator and other state. The execution call gathers all tracks with enough data (audio and emotion) to run an execution and batches them together.

Note that when not all tracks are ready, the Audio2Emotion executor only runs an execution of the minimal batch size corresponding to the number of ready tracks. However, the Audio2Face executors always perform inference of the full batch size as allocated at executor creation time. This is because Audio2Face executors are optimized for the case where batches are full and they contain state which might be expensive to move around for partial batches. This could still be optimized for more efficient execution in the case of partially full batches.

### Executors

Executors are designed to provide the minimal API to process audio.

First, audio is passed to the executor using an [audio accumulator](#audio-accumulator). Optionally, emotions can also be passed using an [emotion accumulator](#emotion-accumulator).

Then, execution can happen if there is enough data. Executors can be queried as to whether or not they have enough data on a given track using `nva2x::IExecutor::GetNbAvailableExecutions()` or for any track using `nva2x::GetNbReadyTracks()`. Execution can be triggered if enough data is available.

Finally, results are reported using a callback mechanism. GPU results are reported using `nva2x::DeviceTensorConstView` objects, which are the equivalent of `std::span` for GPU memory, and an associated CUDA stream. This is because the GPU might not be done with the asynchronous computation at the time the callback is received. This design choice makes it easier to keep the GPU as busy as possible and allows opportunities for maximum concurrency between CPU and GPU.

Executors that return GPU results all behave the same way when it comes to callbacks: all callbacks will be triggered from the same thread that made the call to `nva2x::IExecutor::Execute()` before it returns. These executors include the Audio2Emotion executors, the Audio2Face geometry executors, and the GPU blendshape solve executor. The registered callbacks are called once for each track and for each frame generated by the `nva2x::IExecutor::Execute()` call.

#### Geometry Executors

Audio2Face deep learning models generate facial animation represented as moving vertices from a mesh with known topology. Executors that provide this geometry output are referred to as geometry executors.

#### Blendshape Executors

Audio2Face also provides a different type of executors that output blendshape weights, or rig controls, for facial animation. The only blendshape executor implementation currently available uses a blendshape solver to derive the weights.

The blendshape solve is done on top of the geometry output from a geometry executor. The solver tries to take an existing blendshape rig and find the weights that best match the output from geometry executors.

There are two blendshape solve versions available in the SDK. The GPU blendshape solve implementation uses the GPU to run the necessary computations. This implementation behaves very much like the geometry executor: it receives the not-necessarily-evaluated-yet geometry results and just schedules the following blendshape solve computation on the GPU. It behaves like geometry executors in that all callbacks are done from the same thread that made the call to `nva2x::IExecutor::Execute()` before it returns, right after the blendshape solve work has been scheduled on the GPU, using the same stream mechanism. If results are needed, mechanisms to synchronize with the provided CUDA stream can be used to ensure data is ready, synchronously or asynchronously.

The CPU blendshape solve works differently. When the geometry results are produced by geometry executors, CPU blendshape solve tasks are scheduled on a user-controllable thread pool. The `nva2f::IJobRunner` interface allows control over thread management in the host application, but if not provided, a simple thread pool based on `std::thread` is used instead.

As a result, the call to `nva2x::IExecutor::Execute()` can return before any result callback has been called. The callbacks are triggered from the worker threads in the pool after the work is done. Therefore, after a call to `nva2x::IExecutor::Execute()`, if one needs to make sure all callbacks have been called, `nva2f::IBlendshapeExecutor::Wait()` can be used to wait until all scheduled CPU tasks are done.

### Error Reporting

The Audio2X SDK uses `std::error_code` for consistent and comprehensive error handling across all components. This approach provides type-safe error reporting with detailed error messages and categorization.

The SDK defines custom error categories and codes to provide specific information about different types of failures:

- `nva2x::ErrorCode` in `audio2x-common/include/audio2x/error.h`
- `nva2e::ErrorCode` in `audio2emotion-sdk/include/audio2emotion/error.h`
- `nva2f::ErrorCode` in `audio2face-sdk/include/audio2face/error.h`

#### Basic Error Checking

By default, detected errors and additional information are logged to `stderr`. This behavior is controlled by the `AUDIO2X_LOG_LEVEL`, `AUDIO2EMOTION_LOG_LEVEL`, and `AUDIO2FACE_LOG_LEVEL` compile-time constants, which can be used to control the level of information being logged.

It is also easy to programmatically check for errors. The SDK provides human-readable error messages for all error codes:

```cpp
// Check for errors
std::error_code error = someFunction();
if (error) {
    std::cerr << "Error occurred: " << error.message() << std::endl;
    std::cerr << "Error code: " << error.value() << std::endl;
    std::cerr << "Error category: " << error.category().name() << std::endl;
}
```

These messages might not contain all the information logged to `stderr`, such as failing file paths or additional contextual information about the error.

### ABI Compatibility

The Audio2X SDK is designed with ABI (Application Binary Interface) compatibility in mind to ensure that applications built with different compilers or compiler versions can seamlessly use the SDK. This is achieved through several key design principles:

#### Interface-Based Design

The SDK uses abstract interfaces throughout its API to provide a stable binary interface. All major components are exposed through pure virtual interfaces (e.g., `nva2x::IExecutor`, `nva2x::IAudioAccumulator`, `nva2f::IBlendshapeSolver`), which ensures that the internal implementation details remain hidden from client applications.

#### STL Type Avoidance

The SDK carefully avoids exposing STL types in its public API where the ABI is not guaranteed between different compilers or compiler versions. This includes containers like `std::vector`, `std::string`, `std::map`, etc., which can have different memory layouts and implementations across different STL versions.

The notable exception is `std::error_code`, which is used for error reporting throughout the SDK. This type is chosen because it provides a stable ABI and is essential for comprehensive error handling across all components.

#### Object Creation and Destruction

To ensure proper memory management across different compiler runtimes, the SDK follows a specific pattern for object lifecycle management:

- **Creation**: Factory functions return newly created instances of interfaces. These functions are exported from the SDK and handle the allocation internally.
- **Destruction**: Instead of using the `delete` operator directly, all interface objects expose a `Destroy()` method. This ensures that deletion happens correctly even if the compiler or runtime differs between the SDK and the client application.

Example usage pattern:
```cpp
// Create a CUDA stream
nva2x::ICudaStream* cudaStream = nva2x::CreateCudaStream();

// Use the CUDA stream.
cudaStream->Synchronize();

// Destroy using the SDK's method, not delete
cudaStream->Destroy();
cudaStream = nullptr;
```

This approach prevents issues that could arise from mismatched allocators or different memory layouts between the SDK and client application, ensuring robust binary compatibility across different build environments.

Smart pointers can be used to handle SDK objects lifetime.  For example:
```cpp
struct Destroyer {
    template <typename T>
    void operator()(T* obj) const { obj->Destroy(); }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, Destroyer>;

template <typename T>
UniquePtr<T> ToUniquePtr(T* ptr) { return UniquePtr<T>(ptr); }

template <typename T>
using SharedPtr = std::shared_ptr<T>;

template <typename T>
std::shared_ptr<T> ToSharedPtr(T* ptr) {
  return std::shared_ptr<T>(ptr, [](T* p) { p->Destroy(); });
}

auto uniqueCudaStream = ToUniquePtr(nva2x::CreateCudaStream());
auto sharedCudaStream = ToSharedPtr(nva2x::CreateCudaStream());
```

It is recommended using such smart pointers to safely handle SDK objects.

---

## Models Breakdown

### Audio2Emotion

The Audio2Emotion classifier model is distributed as an ONNX file which can be converted to a TensorRT model. Default parameters for the TensorRT conversion are provided in `trt_info.json`.

The model also comes with an associated `network_info.json` file which contains:

- The model audio input expected sampling rate: 16000 Hz
- The emotion names detected by the network: angry, disgust, fear, happy, neutral, sad

Note that these emotions are the ones produced by the neural network, but they generally need to be remapped before being fed into Audio2Face which supports a different number of emotions.

This is done through a post-processing stage which also remaps emotions from the inference output to other indices such as Audio2Face's. `model_config.json` holds parameters for this post-processing stage, along with how to remap detected emotions.

The Audio2Emotion model works by processing an audio samples window to detect the emotion at the center of this window.  Therefore, it needs available samples for at least half the inference window size ahead of the time for which emotions need to be detected. The inference window size can be configured at runtime, but must be within the limits of what was specified when the TensorRT model was generated based on the dynamic shape values passed for the `input_values` tensor. Inference window management, i.e., extraction of the audio samples surrounding a given frame of interest, can be done manually using the low-level API, but it is generally much simpler and more reliable to provide the desired frame rate to the executor which will handle inference window progress to provide frames at the right timestamps.

### Audio2Face

The Audio2Face SDK generates facial animations from audio input using deep learning models.

Like Audio2Emotion, models are distributed as ONNX files which can be converted to TensorRT models. Default parameters for the TensorRT conversion are provided in `trt_info.json`.

It uses two main approaches:

#### Regression-based Animation

Like Audio2Emotion's classifier model, the Audio2Face regression model provides facial animation at the center of the audio sample window provided to the model. Therefore, it provides random-access in the audio buffer and can produce frames for any frame rate. The inference buffer size is given by the `buffer_len` parameter in `network_info.json`.

It also takes emotion as an input. Part of the emotion parameter consists of _implicit_ emotions which come from an emotion database, associated with a specific frame within a specific shot. These are specified as the `source_frame` and `source_shot` parameters of `model_config.json` and should not be tampered with. The other part consists of _explicit_ emotions which are intended to guide the facial performance. These emotions should either be provided by the user to match the intent or can come straight from Audio2Emotion.

Geometry output for regression models is provided as PCA coefficients. PCA reconstruction is done by applying those coefficients to shape matrices stored in `model_data.npz`. The output of the reconstruction is vertex position deltas with respect to a neutral pose. The neutral pose could be added as part of the PCA reconstruction, but it is not because geometry post-processing is done in the delta space, and therefore the neutral pose is added after post-processing.

#### Diffusion-based Animation

The Audio2Face diffusion model works differently. It computes many frames of animation within a single inference. The actual numbers are specified in `network_info.json`, but as an example:

- For a single second of audio samples (`buffer_len`), many frames are generated.
- 15 (`num_frames_left_truncate`) frames are generated before "good" frames and meant to be discarded.
- 15 (`num_frames_right_truncate`) frames are generated after "good" frames and meant to be discarded.
- 30 (`num_frames_center`) frames are generated and meant to be used.

This model generates 60 frames per second, but half of them are discarded. Therefore, each inference is distanced by only half a second step so that good frames are really generated one after the other. Note that emotion data must be available for all the good frames in order to run an inference.

This also means that calls to `nva2x::IExecutor::Execute()` can generate up to 30 callbacks per track. The number can be lower because execution actually starts _before_ real audio is processed. This is controlled by `padding_left` and is used to warm up the inference. The number can also be lower if the generated frames go past the end of the audio. Diffusion model executors will not generate frames before the beginning or after the end of the audio track, even though those frames are actually computed.

The output of the diffusion model is vertex position deltas with respect to the neutral pose. Therefore, the diffusion model does not need PCA reconstruction.

Another distinction for diffusion is that the same model supports multiple identities. This allows different identities to learn from one another and potentially use data from one identity to help another one fill a gap in their training data. As a result, an extra identity index must be provided to select which one is to be used. Available identities are listed in the `identities` field in `network_info.json`.

The diffusion model maintains state between each inference.  Unlike the regression model, each inference must be run sequentially one after the other in a contiguous way, where the state of the previous inference must be passed to the next.

#### Post-processing

Both regression and diffusion models provide the same output to which similar post-processing is applied.

Skin geometry (PCA coefficients converted to deltas for regression model and direct deltas for diffusion model) is fed to a processing stage which provides control over smoothing, movement strength, etc., controllable for the upper or lower parts of the face.

Tongue geometry (also coming from PCA for regression and deltas for diffusion) provides fewer controls for post-processing, mostly strength and offsets.

Teeth output is actually a set of anchor points from which a rigid (rotation + translation) transformation is extracted with respect to the neutral pose. This can be useful to drive the transform in a rig controlling teeth position, for example.

Eyes output is horizontal and vertical rotation for each eye. Post-processing includes strength and offsets to apply to those values, but also the application of a saccade behavior, which provides more natural eye movement.

---

## Conclusion

The Audio2X SDK provides a comprehensive solution for audio-driven animation and emotion detection. With its modular architecture, high-performance GPU acceleration, and flexible API, it's suitable for a wide range of applications from real-time interactive systems to offline content creation.

For more detailed information about specific components, refer to the individual SDK documentation and sample code provided in the SDK distribution.