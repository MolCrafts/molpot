#include "scatter/cpu/segment_coo_cpu.h"

#include "scatter/cpu/index_info.h"
#include "scatter/cpu/reducer.h"
#include "scatter/cpu/utils.h"
#include <ATen/OpMathType.h>

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
segment_coo_cpu(torch::Tensor src, torch::Tensor index,
                std::optional<torch::Tensor> optional_out,
                std::optional<int64_t> dim_size, std::string reduce) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  if (optional_out.has_value())
    CHECK_CPU(optional_out.value());

  CHECK_INPUT(src.dim() >= index.dim());

  auto sizes = index.sizes().vec();
  for (auto i = 0; i < index.dim(); i++)
    sizes[i] = src.size(i);
  index = index.expand(sizes);

  auto dim = index.dim() - 1;

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (auto i = 0; i < out.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
  } else {
    sizes = src.sizes().vec();
    if (dim_size.has_value())
      sizes[dim] = dim_size.value();
    else if (index.numel() == 0)
      sizes[dim] = 0;
    else {
      auto tmp = index.select(dim, index.size(dim) - 1);
      tmp = tmp.numel() > 1 ? tmp.max() : tmp;
      sizes[dim] = 1 + *tmp.data_ptr<int64_t>();
    }
    out = torch::empty(sizes, src.options());
  }

  std::optional<torch::Tensor> arg_out = std::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, src.size(dim), index.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  } else if (reduce2REDUCE.at(reduce) == MEAN) {
    auto sizes = index.sizes().vec();
    sizes[dim] = out.size(dim);
    arg_out = torch::zeros(sizes, out.options());
  }

  if (src.numel() == 0) {
    if (!optional_out.has_value())
      out.fill_(0);
    return std::make_tuple(out, arg_out);
  }

  auto B = index.numel() / src.size(dim);
  auto E = src.size(dim);
  auto K = src.numel() / index.numel();
  auto N = out.size(dim);

  auto index_info = getTensorInfo<int64_t>(index);
  auto stride = index_info.strides[index_info.dims - 1];
  std::vector<int64_t> args(K);
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "segment_coo_cpu", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    scalar_t *count_data = nullptr;

    std::vector<opmath_t> vals(K);
    int64_t idx, next_idx, row_start;
    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (!optional_out.has_value())
        out.fill_(Reducer<scalar_t, REDUCE>::init());
      if (REDUCE == MEAN)
        count_data = arg_out.value().data_ptr<scalar_t>();

      for (auto b = 0; b < B; b++) {
        auto offset = IndexToOffset<int64_t>::get(b * E, index_info);
        idx = index_info.data[offset];

        for (auto k = 0; k < K; k++)
          vals[k] = static_cast<opmath_t>(out_data[b * N * K + k]);

        row_start = 0;
        for (auto e = 0; e < E; e++) {

          for (auto k = 0; k < K; k++)
            Reducer<opmath_t, REDUCE>::update(
                &vals[k], static_cast<opmath_t>(src_data[b * E * K + e * K + k]), &args[k], e);

          if (e == E - 1) {
            for (auto k = 0; k < K; k++)
              Reducer<scalar_t, REDUCE>::write(
                  out_data + b * N * K + idx * K + k, static_cast<scalar_t>(vals[k]),
                  arg_out_data + b * N * K + idx * K + k, args[k],
                  e + 1 - row_start);
            if (REDUCE == MEAN)
              count_data[b * N + idx] = (scalar_t)(e + 1 - row_start);
          } else {
            next_idx = index_info.data[offset + (e + 1) * stride];
            assert(idx <= next_idx);

            if (idx != next_idx) {
              for (auto k = 0; k < K; k++) {
                Reducer<scalar_t, REDUCE>::write(
                    out_data + b * N * K + idx * K + k, static_cast<scalar_t>(vals[k]),
                    arg_out_data + b * N * K + idx * K + k, args[k],
                    e + 1 - row_start);

                vals[k] = static_cast<opmath_t>(out_data[b * N * K + next_idx * K + k]);
              }
              if (REDUCE == MEAN)
                count_data[b * N + idx] = (scalar_t)(e + 1 - row_start);
              row_start = e + 1;
            }

            idx = next_idx;
          }
        }
      }
      if (!optional_out.has_value() && (REDUCE == MIN || REDUCE == MAX))
        out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);

      if (REDUCE == MEAN)
        arg_out.value().masked_fill_(arg_out.value() < (scalar_t)1,
                                     (scalar_t)1);
    });
  });

  return std::make_tuple(out, arg_out);
}

torch::Tensor gather_coo_cpu(torch::Tensor src, torch::Tensor index,
                             std::optional<torch::Tensor> optional_out) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  if (optional_out.has_value())
    CHECK_CPU(optional_out.value());

  CHECK_INPUT(src.dim() >= index.dim());
  for (auto i = 0; i < index.dim() - 1; i++)
    CHECK_INPUT(src.size(i) == index.size(i));

  auto dim = index.dim() - 1;

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (auto i = 0; i < src.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
  } else {
    auto sizes = src.sizes().vec();
    sizes[dim] = index.size(dim);
    out = torch::empty(sizes, src.options());
  }

  if (src.numel() == 0) {
    if (!optional_out.has_value())
      out.fill_(0);
    return out;
  }

  auto B = index.numel() / out.size(dim);
  auto E = index.size(dim);
  auto K = out.numel() / index.numel();
  auto N = src.size(dim);

  auto index_info = getTensorInfo<int64_t>(index);
  auto stride = index_info.strides[index_info.dims - 1];
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "gather_coo_cpu", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    std::vector<scalar_t> vals(K);
    int64_t idx, next_idx;
    for (auto b = 0; b < B; b++) {
      auto offset = IndexToOffset<int64_t>::get(b * E, index_info);
      idx = index_info.data[offset];

      for (auto k = 0; k < K; k++)
        vals[k] = src_data[b * N * K + idx * K + k];

      for (auto e = 0; e < E; e++) {
        for (auto k = 0; k < K; k++)
          out_data[b * E * K + e * K + k] = vals[k];

        if (e < E - 1) {
          next_idx = index_info.data[offset + (e + 1) * stride];
          CHECK_INPUT(idx <= next_idx);

          if (idx != next_idx) {
            idx = next_idx;
            for (auto k = 0; k < K; k++)
              vals[k] = src_data[b * N * K + idx * K + k];
          }
        }
      }
    }
  });

  return out;
}
