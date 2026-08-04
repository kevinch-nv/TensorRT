#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import io
import os
import yaml


class ReqYaml:

    args = {}
    conditions = {}
    packages = {}

    def __init__(self, req_yaml_path):
        assert os.path.isfile(req_yaml_path)
        with open(req_yaml_path, "r") as req_yaml_file:
            yaml_python_map = yaml.safe_load(req_yaml_file)
            self.args = {} if not yaml_python_map["args"] else yaml_python_map["args"]
            self.conditions = (
                {}
                if not yaml_python_map["conditions"]
                else yaml_python_map["conditions"]
            )
            self.packages = (
                {} if not yaml_python_map["packages"] else yaml_python_map["packages"]
            )

    def inherit_from(self, top_req_yaml):
        top_conditions = top_req_yaml.conditions
        top_args = top_req_yaml.args
        top_packages = top_req_yaml.packages

        for package in top_conditions:
            if package not in self.conditions:
                self.conditions[package] = top_conditions[package]

        for package in top_args:
            if package not in self.args:
                self.args[package] = top_args[package]

        for package in top_packages:
            if package not in self.packages:
                self.packages.append(package)

    def gen_requirements_txt(self):
        buf = io.StringIO()
        for package in self.packages:
            if package in self.args:
                for arg in self.args[package]:
                    buf.write(f"{arg}\n")
            if package in self.conditions:
                for condition in self.conditions[package]:
                    buf.write(f"{condition}\n")
            else:
                buf.write(f"{package}\n")

        buf.seek(0)
        return buf


if __name__ == "__main__":
    import glob

    directory = os.path.dirname(os.path.realpath(__file__))

    top_req_yaml_path = os.path.join(directory, "requirements.yml")

    top_req = None
    if os.path.isfile(top_req_yaml_path):
        top_req = ReqYaml(top_req_yaml_path)
        req_buf = top_req.gen_requirements_txt()
        with open(os.path.join(directory, "requirements.txt"), mode="w") as f:
            f.write(req_buf.getvalue())

    for req_yaml_path in glob.glob(os.path.join(directory, "*/requirements.yml")):
        req = ReqYaml(req_yaml_path)
        if top_req is not None:
            req.inherit_from(top_req)
        req_buf = req.gen_requirements_txt()
        sample_dir = os.path.dirname(os.path.realpath(req_yaml_path))
        with open(os.path.join(sample_dir, "requirements.txt"), mode="w") as f:
            f.write(req_buf.getvalue())
