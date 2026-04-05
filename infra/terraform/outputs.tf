output "ec2_public_ip" {
  value = aws_instance.trainer.public_ip
}

output "ec2_instance_id" {
  value = aws_instance.trainer.id
}

output "artifacts_bucket" {
  value = local.resolved_artifacts_bucket
}
