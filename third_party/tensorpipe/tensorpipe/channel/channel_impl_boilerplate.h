/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <utility>

#include <tensorpipe/channel/channel.h>
#include <tensorpipe/channel/error.h>
#include <tensorpipe/common/callback.h>
#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/error.h>
#include <tensorpipe/common/error_macros.h>

namespace tensorpipe {
namespace channel {

template <typename TBuffer, typename TCtx, typename TChan>
class ContextImplBoilerplate;

template <typename TBuffer, typename TCtx, typename TChan>
class ChannelImplBoilerplate : public std::enable_shared_from_this<TChan> {
 public:
  class ConstructorToken {
   public:
    ConstructorToken(const ConstructorToken&) = default;

   private:
    explicit ConstructorToken() {}
    friend ContextImplBoilerplate<TBuffer, TCtx, TChan>;
  };

  ChannelImplBoilerplate(
      ConstructorToken token,
      std::shared_ptr<TCtx> context,
      std::string id);

  ChannelImplBoilerplate(const ChannelImplBoilerplate&) = delete;
  ChannelImplBoilerplate(ChannelImplBoilerplate&&) = delete;
  ChannelImplBoilerplate& operator=(const ChannelImplBoilerplate&) = delete;
  ChannelImplBoilerplate& operator=(ChannelImplBoilerplate&&) = delete;

  // Initialize member fields that need `shared_from_this`.
  void init();

  // Perform a send operation.
  void send(
      TBuffer buffer,
      TDescriptorCallback descriptorCallback,
      TSendCallback callback);

  // Queue a recv operation.
  void recv(TDescriptor descriptor, TBuffer buffer, TRecvCallback callback);

  // Tell the connection what its identifier is.
  void setId(std::string id);

  // Shut down the connection and its resources.
  void close();

  virtual ~ChannelImplBoilerplate() = default;

 protected:
  virtual void initImplFromLoop() = 0;
  virtual void sendImplFromLoop(
      uint64_t sequenceNumber,
      TBuffer buffer,
      TDescriptorCallback descriptorCallback,
      TSendCallback callback) = 0;
  virtual void recvImplFromLoop(
      uint64_t sequenceNumber,
      TDescriptor descriptor,
      TBuffer buffer,
      TRecvCallback callback) = 0;
  virtual void handleErrorImpl() = 0;
  virtual void setIdImpl() {}

  void setError(Error error);

  const std::shared_ptr<TCtx> context_;

  Error error_{Error::kSuccess};

  // An identifier for the connection, composed of the identifier for the
  // context or listener, combined with an increasing sequence number. It will
  // only be used for logging and debugging purposes.
  std::string id_;

  LazyCallbackWrapper<TChan> lazyCallbackWrapper_{*this, *this->context_};
  EagerCallbackWrapper<TChan> eagerCallbackWrapper_{*this, *this->context_};

 private:
  // Initialize member fields that need `shared_from_this`.
  void initFromLoop();

  // Perform a send operation.
  void sendFromLoop(
      TBuffer buffer,
      TDescriptorCallback descriptorCallback,
      TSendCallback callback);

  // Queue a recv operation.
  void recvFromLoop(
      TDescriptor descriptor,
      TBuffer buffer,
      TRecvCallback callback);

  void setIdFromLoop(std::string id);

  // Shut down the connection and its resources.
  void closeFromLoop();

  // Deal with an error.
  void handleError();

  // A sequence number for the calls to send and recv.
  uint64_t nextTensorBeingSent_{0};
  uint64_t nextTensorBeingReceived_{0};

  // For some odd reason it seems we need to use a qualified name here...
  template <typename T>
  friend class tensorpipe::LazyCallbackWrapper;
  template <typename T>
  friend class tensorpipe::EagerCallbackWrapper;

  // Contexts do sometimes need to call directly into closeFromLoop, in order to
  // make sure that some of their operations can happen "atomically" on the
  // connection, without possibly other operations occurring in between (e.g.,
  // an error).
  friend ContextImplBoilerplate<TBuffer, TCtx, TChan>;
};

template <typename TBuffer, typename TCtx, typename TChan>
ChannelImplBoilerplate<TBuffer, TCtx, TChan>::ChannelImplBoilerplate(
    ConstructorToken /* unused */,
    std::shared_ptr<TCtx> context,
    std::string id)
    : context_(std::move(context)), id_(std::move(id)) {}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::init() {
  context_->deferToLoop(
      [impl{this->shared_from_this()}]() { impl->initFromLoop(); });
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::initFromLoop() {
  if (context_->closed()) {
    // Set the error without calling setError because we do not want to invoke
    // the subclass's handleErrorImpl as it would find itself in a weird state
    // (since initFromLoop wouldn't have been called).
    error_ = TP_CREATE_ERROR(ChannelClosedError);
    TP_VLOG(4) << "Channel " << id_ << " is closing (without initing)";
    return;
  }

  initImplFromLoop();
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::send(
    TBuffer buffer,
    TDescriptorCallback descriptorCallback,
    TSendCallback callback) {
  context_->deferToLoop([impl{this->shared_from_this()},
                         buffer,
                         descriptorCallback{std::move(descriptorCallback)},
                         callback{std::move(callback)}]() mutable {
    impl->sendFromLoop(
        buffer, std::move(descriptorCallback), std::move(callback));
  });
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::sendFromLoop(
    TBuffer buffer,
    TDescriptorCallback descriptorCallback,
    TSendCallback callback) {
  TP_DCHECK(context_->inLoop());

  const uint64_t sequenceNumber = nextTensorBeingSent_++;
  TP_VLOG(4) << "Channel " << id_ << " received a send request (#"
             << sequenceNumber << ")";

  descriptorCallback = [this,
                        sequenceNumber,
                        descriptorCallback{std::move(descriptorCallback)}](
                           const Error& error, TDescriptor descriptor) {
    // There is no requirement for the channel to invoke callbacks in order.
    TP_VLOG(4) << "Channel " << id_ << " is calling a descriptor callback (#"
               << sequenceNumber << ")";
    descriptorCallback(error, std::move(descriptor));
    TP_VLOG(4) << "Channel " << id_ << " done calling a descriptor callback (#"
               << sequenceNumber << ")";
  };

  callback = [this, sequenceNumber, callback{std::move(callback)}](
                 const Error& error) {
    // There is no requirement for the channel to invoke callbacks in order.
    TP_VLOG(4) << "Channel " << id_ << " is calling a send callback (#"
               << sequenceNumber << ")";
    callback(error);
    TP_VLOG(4) << "Channel " << id_ << " done calling a send callback (#"
               << sequenceNumber << ")";
  };

  if (error_) {
    descriptorCallback(error_, std::string());
    callback(error_);
    return;
  }

  sendImplFromLoop(
      sequenceNumber,
      buffer,
      std::move(descriptorCallback),
      std::move(callback));
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::recv(
    TDescriptor descriptor,
    TBuffer buffer,
    TRecvCallback callback) {
  context_->deferToLoop([impl{this->shared_from_this()},
                         descriptor{std::move(descriptor)},
                         buffer,
                         callback{std::move(callback)}]() mutable {
    impl->recvFromLoop(std::move(descriptor), buffer, std::move(callback));
  });
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::recvFromLoop(
    TDescriptor descriptor,
    TBuffer buffer,
    TRecvCallback callback) {
  TP_DCHECK(context_->inLoop());

  const uint64_t sequenceNumber = nextTensorBeingReceived_++;
  TP_VLOG(4) << "Channel " << id_ << " received a recv request (#"
             << sequenceNumber << ")";

  callback = [this, sequenceNumber, callback{std::move(callback)}](
                 const Error& error) {
    // There is no requirement for the channel to invoke callbacks in order.
    TP_VLOG(4) << "Channel " << id_ << " is calling a recv callback (#"
               << sequenceNumber << ")";
    callback(error);
    TP_VLOG(4) << "Channel " << id_ << " done calling a recv callback (#"
               << sequenceNumber << ")";
  };

  if (error_) {
    callback(error_);
    return;
  }

  recvImplFromLoop(
      sequenceNumber, std::move(descriptor), buffer, std::move(callback));
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::setId(std::string id) {
  context_->deferToLoop(
      [impl{this->shared_from_this()}, id{std::move(id)}]() mutable {
        impl->setIdFromLoop(std::move(id));
      });
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::setIdFromLoop(
    std::string id) {
  TP_DCHECK(context_->inLoop());
  TP_VLOG(4) << "Channel " << id_ << " was renamed to " << id;
  id_ = std::move(id);
  setIdImpl();
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::close() {
  context_->deferToLoop(
      [impl{this->shared_from_this()}]() { impl->closeFromLoop(); });
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::closeFromLoop() {
  TP_DCHECK(context_->inLoop());
  TP_VLOG(4) << "Channel " << id_ << " is closing";
  setError(TP_CREATE_ERROR(ChannelClosedError));
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::setError(Error error) {
  // Don't overwrite an error that's already set.
  if (error_ || !error) {
    return;
  }

  error_ = std::move(error);

  handleError();
}

template <typename TBuffer, typename TCtx, typename TChan>
void ChannelImplBoilerplate<TBuffer, TCtx, TChan>::handleError() {
  TP_DCHECK(context_->inLoop());
  TP_VLOG(5) << "Channel " << id_ << " is handling error " << error_.what();

  handleErrorImpl();
}

} // namespace channel
} // namespace tensorpipe
