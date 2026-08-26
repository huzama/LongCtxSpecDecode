# SPDX-License-Identifier: Apache-2.0
"""Kernel-independent strategies for the two features vegas takes from FA3.

Score collection and draft KV addressing each come in a kernel-native and a
portable form; ``kernel_support`` picks by what the loaded binary offers.
"""
