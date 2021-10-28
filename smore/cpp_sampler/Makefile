# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

dir_guard = @mkdir -p $(@D)
FIND := find
CXX := g++

CXXFLAGS += -Wall -O3 -std=c++11 -Wfatal-errors
LDFLAGS += -lm

CXXFLAGS += $(addprefix -I,$(include_dirs))
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	PY_FLAG := -undefined dynamic_lookup
	CXXFLAGS += -Wno-unused-local-typedef
else
	PY_FLAG := 
endif


include_dirs = ./include ../../third_party/ThreadPool

CXXFLAGS += -fPIC
PYBIND = $(shell python3 -m pybind11 --includes)
cpp_files = $(shell $(FIND) src/lib -name "*.cpp" -print | rev | cut -d"/" -f1 | rev)
cxx_obj_files = $(subst .cpp,.o,$(cpp_files))

objs = $(addprefix build/lib/,$(cxx_obj_files))
DEPS = $(objs:.o=.d)

target = build/dll/libsampler.so build/dll/libsampler_c.so
target_dep = $(addsuffix .d,$(target))

.PRECIOUS: build/lib/%.o

all: $(target)

build/dll/libsampler.so : src/pybind.cpp $(objs)
	$(dir_guard)
	$(CXX) -shared $(CXXFLAGS) $(PY_FLAG) $(PYBIND) -MMD -o $@ $(filter %.cpp %.o, $^) $(LDFLAGS)

build/dll/libsampler_c.so : src/sampler_clib.cpp $(objs)
	$(dir_guard)
	$(CXX) -shared $(CXXFLAGS) $(PY_FLAG) $(PYBIND) -MMD -o $@ $(filter %.cpp %.o, $^) $(LDFLAGS)

DEPS += $(target_dep)

build/lib/%.o: src/lib/%.cpp
	$(dir_guard)
	$(CXX) $(CXXFLAGS) $(PYBIND) -MMD -c -o $@ $(filter %.cpp, $^)

clean:
	rm -rf build

-include $(DEPS)
