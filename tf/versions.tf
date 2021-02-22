# Configure the Azure provider
terraform {
  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
      version = ">= 2.0"
    }
    azuread = {
      source = "hashicorp/azuread"
      version = ">= 0.7.0"
    }
  }
}