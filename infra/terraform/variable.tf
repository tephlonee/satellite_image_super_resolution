variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "SAR"
}

variable "github_repo_url" {
  type    = string
  default = "https://github.com/tephlonee/satellite_image_super_resolution.git"
}

variable "github_ref" {
  type    = string
  default = "feature_changes"
}

variable "instance_type" {
  type    = string
  default = "g5.xlarge"
}

variable "use_spot" {
  type    = bool
  default = true
}

variable "spot_max_price" {
  type    = string
  default = ""
}

variable "key_name" {
  type    = string
  default = null
}

variable "allowed_ssh_cidr" {
  type    = string
  default = "0.0.0.0/0"
}

variable "ami_id" {
  type    = string
  default = null
}

variable "dlami_ssm_parameter_name" {
  type    = string
  default = "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id"
}

variable "vpc_id" {
  type    = string
  default = null
}

variable "subnet_id" {
  type    = string
  default = null
}

variable "data_volume_size_gb" {
  type    = number
  default = 250
}

variable "data_root" {
  type    = string
  default = "/mnt/ebs"
}

variable "data_subdir" {
  type    = string
  default = "batch_downloads"
}

variable "dataset_s3_uri" {
  type    = string
  default = ""
}

variable "s3_no_sign_request" {
  type    = bool
  default = false
}

variable "artifacts_bucket" {
  type    = string
  default = ""
}

variable "artifacts_bucket_prefix" {
  type    = string
  default = "sar-sr-artifacts-"
}

variable "runs_prefix" {
  type    = string
  default = "runs"
}

variable "dataset_bucket" {
  type    = string
  default = ""
}

variable "dataset_prefix" {
  type    = string
  default = ""
}
