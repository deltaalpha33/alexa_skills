/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * main.c
 * Copyright (C) 2017 
 * 
 */

#include <stdio.h>
int main()
{
	char str[999];
	FILE * file;
	file = fopen( "test.txt" , "r");
	if (file) {
		while (fscanf(file, "%s", str)!=EOF)
		    printf("%s",str);
		fclose(file);
	}
	return (0);
}

